"""Trace a literal Hydra recipe from a committed YAML mapping to its optimizer."""

import ast
import re
from pathlib import PurePosixPath

import yaml

from .flag_lr import Sources, chain
from .initial_lr import NUMBER, OPTIMIZERS, main_body, normalized, straight_line_calls


IDENT = re.compile(r"[A-Za-z_]\w*")
COMPONENT = re.compile(r"[\w-]+")


def scalar(node):
    return node.value if isinstance(node, yaml.ScalarNode) else None


def mapping(node):
    if not isinstance(node, yaml.MappingNode) or node.tag != "tag:yaml.org,2002:map":
        raise ValueError("Hydra requires a literal YAML mapping.")
    pairs = {}
    for key, value in node.value:
        name = scalar(key)
        if not name or name in pairs or key.tag != "tag:yaml.org,2002:str" or name == "<<":
            raise ValueError("Unsupported YAML mapping key, merge or duplicate.")
        pairs[name] = value
    return pairs


def yaml_mapping(snapshot, path):
    if path not in snapshot.files:
        raise ValueError(f"The selected Hydra configuration is absent: {path}.")
    text = snapshot.read(path)
    try:
        if re.search(r"(?m)^\s*#\s*@package\b", text) or any(
            isinstance(token, (yaml.AliasToken, yaml.AnchorToken, yaml.TagToken)) for token in yaml.scan(text)
        ):
            raise ValueError("Hydra package directives, YAML aliases and explicit tags are unsupported.")
        return mapping(yaml.compose(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as error:
        raise ValueError(f"Cannot parse Hydra YAML at {path}.") from error


def yaml_site(snapshot, path, node):
    start, end = node.start_mark.line, node.end_mark.line
    return {"path": path, "line_start": start + 1, "line_end": end + 1,
            "excerpt": "\n".join(snapshot.read(path).splitlines()[start:end + 1]).strip()}


def compose(snapshot, path, overrides, candidate):
    definitions, trail = candidate["definitions"], candidate["stage_trace"]
    base = yaml_mapping(snapshot, path)
    default_lr = base.get("lr")
    if default_lr:
        definitions.append(yaml_site(snapshot, path, default_lr))
    defaults = base.pop("defaults", None)
    if defaults is not None and not isinstance(defaults, yaml.SequenceNode):
        raise ValueError("Hydra defaults must be a list of flat, named configuration groups.")
    groups = {}
    has_self = False
    for item in defaults.value if defaults is not None else []:
        if scalar(item) == "_self_" and not has_self:
            has_self = True
            continue
        group = mapping(item)
        if len(group) != 1:
            raise ValueError("Unsupported Hydra defaults entry.")
        name, selected = next(iter(group.items()))
        if not IDENT.fullmatch(name) or name in groups or name in base:
            raise ValueError("Unsupported or overlapping Hydra configuration group.")
        choice = overrides.get(name, scalar(selected))
        if not choice or not COMPONENT.fullmatch(choice):
            raise ValueError("The selected Hydra group is not literal.")
        group_path = str(PurePosixPath(path).parent / name / f"{choice}.yaml")
        fields = yaml_mapping(snapshot, group_path)
        if "defaults" in fields:
            raise ValueError("Nested Hydra defaults are unsupported.")
        groups[name] = (group_path, fields)
        trail.append(yaml_site(snapshot, path, selected))

    resolved = {key: scalar(value) for key, value in base.items()}
    for key, value in overrides.items():
        if key in groups:
            continue
        head, dot, tail = key.partition(".")
        if dot:
            if head not in groups or tail not in groups[head][1] or not isinstance(groups[head][1][tail], yaml.ScalarNode):
                raise ValueError(f"Unsupported Hydra override: {key}.")
        elif head not in base or isinstance(base[head], yaml.MappingNode):
            raise ValueError(f"Unsupported Hydra override: {key}.")
        resolved[key] = value

    if "model" in groups:
        model_path, fields = groups["model"]
        name = fields.get("name_or_path")
        if name is not None:
            trail.append(yaml_site(snapshot, model_path, name))
        model = resolved.get("model.name_or_path", scalar(name))
        if not model or not re.fullmatch(r"[\w./-]+", model):
            raise ValueError("The selected Hydra model identity is not literal.")
        candidate["model_name"] = model

    if "lr" in overrides:
        candidate["command_override"] = overrides["lr"]
    raw = resolved.get("lr")
    if (raw is None or not re.fullmatch(NUMBER, raw) or normalized(raw) is None
            or ("lr" not in overrides and default_lr.style is not None)):
        raise ValueError("The base Hydra LR is missing, dynamic or interpolated.")
    for key in ("optimizer", "trainer"):
        if key in base:
            trail.append(yaml_site(snapshot, path, base[key]))
    return resolved


def unique(items, message):
    if len(items) != 1:
        raise ValueError(message)
    return items[0]


def method(body, name):
    return unique([item for item in body if isinstance(item, ast.FunctionDef) and item.name == name],
                  f"Cannot resolve a unique Python function {name}.")


def formal(fn):
    return [arg.arg for arg in fn.args.posonlyargs + fn.args.args]


def attribute_writes(node, prefix, permitted=()):
    """Detect tracked assignments, dynamic attribute access and namespace escapes."""
    children = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
    for item in ast.walk(node):
        ref = chain(item)
        if ref == prefix:
            parent = children.get(item)
            if item not in permitted and (isinstance(item.ctx, (ast.Store, ast.Del)) or (
                not isinstance(parent, ast.Attribute)
            )):
                yield item
        elif ref[:len(prefix)] == prefix and len(ref) > len(prefix):
            key = ref[len(prefix)]
            parent = children.get(item)
            direct_call = isinstance(parent, ast.Call) and parent.func is item and len(ref) == len(prefix) + 1
            model_write = key == "model" and (len(ref) == len(prefix) + 1 or ref[len(prefix) + 1] == "name_or_path")
            if (direct_call or key.startswith("__") or
                    (key in {"lr", "optimizer", "trainer"} or model_write)
                    and (isinstance(item.ctx, (ast.Store, ast.Del)) or isinstance(parent, ast.Subscript))):
                yield item


def worker_route(body, entry, option):
    names = {node.name for node in body if isinstance(node, ast.FunctionDef)}
    routes = []
    for call in ast.walk(entry):
        if not isinstance(call, ast.Call):
            continue
        callee = chain(call.func)
        args = call.args
        prefix = 0
        if callee == ("mp", "spawn") and args and isinstance(args[0], ast.Name):
            callee = (args[0].id,)
            spawn_args = next((kw.value for kw in call.keywords if kw.arg == "args"), None)
            if not isinstance(spawn_args, ast.Tuple):
                continue
            args, prefix = spawn_args.elts, 1
        if len(callee) == 1 and callee[0] in names:
            for index, arg in enumerate(args, prefix):
                if chain(arg) == (option,):
                    routes.append((callee[0], index, arg, call))
    route = unique(list({(name, idx) for name, idx, _, _ in routes}), "Cannot resolve one Hydra worker dispatch.")
    return method(body, route[0]), route[1], routes


def trainer_use(sources, path, worker, setting, resolved, trail, allowed):
    document = sources.read(path)[1]
    selectors = []
    for node in worker.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        call = node.value
        if (isinstance(call, ast.Call) and chain(call.func) == ("getattr",) and len(call.args) == 2
                and chain(call.args[1]) == (setting, "trainer") and isinstance(call.args[0], ast.Name)):
            selectors.append((node, call.args[0].id))
    selector, module_name = unique(selectors, "Cannot resolve the trainer selected by Hydra.")
    imported = unique([alias.name for node in document if isinstance(node, ast.Import) for alias in node.names
                       if (alias.asname or alias.name) == module_name], "Cannot resolve the trainer module import.")
    trainer_path = sources.module(imported)
    if not trainer_path:
        raise ValueError("The trainer module is absent or ambiguous on the pinned commit.")
    calls = [node for node in worker.body if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
             and chain(node.value.func) == (selector.targets[0].id,)]
    constructor = unique(calls, "Cannot resolve the selected trainer construction.")
    slot, arg = unique([(i, arg) for i, arg in enumerate(constructor.value.args, 1) if chain(arg) == (setting,)],
                       "Cannot trace the Hydra config into the trainer constructor.")
    allowed.append(arg)
    train = unique([call for call in straight_line_calls(worker.body)
                    if chain(call.func) == (constructor.targets[0].id, "train") and not call.args and not call.keywords],
                   "Cannot resolve the call to train the selected trainer.")
    if not selector.lineno < constructor.lineno < train.lineno:
        raise ValueError("Cannot establish the trainer construction and training order.")
    if list(attribute_writes(worker, (setting,), allowed)):
        raise ValueError("The worker may change the Hydra optimizer settings.")
    trail.extend(sources.site(path, node) for node in (selector, constructor, train))

    name = resolved.get("trainer")
    classes = {node.name: node for node in sources.read(trainer_path)[1] if isinstance(node, ast.ClassDef)}
    cls = classes.get(name)
    if not cls:
        raise ValueError("The selected trainer class is not statically resolved.")
    chain_of_classes = [cls]
    while True:
        if len(cls.bases) > 1 or any(not isinstance(base, ast.Name)
                                    or base.id not in {*classes, "object"} for base in cls.bases):
            raise ValueError("Only single local trainer inheritance is supported.")
        local_bases = [classes[base.id] for base in cls.bases if base.id != "object"]
        if not local_bases:
            break
        cls = unique(local_bases, "Multiple trainer bases are unsupported.")
        if cls in chain_of_classes:
            raise ValueError("Cyclic trainer inheritance.")
        chain_of_classes.append(cls)
    for cls in chain_of_classes:
        trail.append({"path": trainer_path, "line_start": cls.lineno, "line_end": cls.lineno,
                      "excerpt": sources.read(trainer_path)[0].splitlines()[cls.lineno - 1].strip()})
    return trainer_path, chain_of_classes, slot


def optimizer_site(sources, path, classes, slot, candidate):
    trail = candidate["stage_trace"]
    owner = next((cls for cls in classes if any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in cls.body)), None)
    if owner is None:
        raise ValueError("The trainer does not define or inherit a local train method.")
    train = method(owner.body, "train")
    self_name = formal(train)[0]
    calls = [call for call in straight_line_calls(train.body)
             if any(kw.arg == "lr" and chain(kw.value) == (self_name, "config", "lr") for kw in call.keywords)]
    call = unique(calls, "Cannot trace a unique straight-line optimizer call using config.lr.")
    candidate["use"] = sources.site(path, call)
    constructor_found = False
    for cls in classes:
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
        if not methods:
            continue
        ctor = unique(methods, "Multiple trainer constructors are unsupported.")
        args = formal(ctor)
        if len(args) <= slot:
            raise ValueError("Cannot bind the trainer's config parameter.")
        receiver = (args[0], "config")
        config = (args[slot],)
        copies = [node for node in ctor.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and chain(node.targets[0]) == receiver and chain(node.value) == config]
        forwards = [call for call in straight_line_calls(ctor.body)
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "__init__"
                    and isinstance(call.func.value, ast.Call) and chain(call.func.value.func) == ("super",)
                    and len(call.args) >= slot and chain(call.args[slot - 1]) == config]
        if not copies and not forwards:
            raise ValueError("Cannot trace the trainer configuration through its constructor.")
        if list(attribute_writes(ctor, config, [*(node.value for node in copies),
                                                *(call.args[slot - 1] for call in forwards)])):
            raise ValueError("The trainer constructor may change the Hydra optimizer settings.")
        if list(attribute_writes(ctor, receiver, [node.targets[0] for node in copies])):
            raise ValueError("The trainer constructor changes its stored configuration.")
        constructor_found |= bool(copies)
        trail.extend(sources.site(path, node) for node in [*copies, *forwards])
    if not constructor_found or list(attribute_writes(train, (self_name, "config"))):
        raise ValueError("Cannot establish an unchanged trainer configuration at optimizer creation.")

    target = call.func
    if not (isinstance(target, ast.Call) and chain(target.func) == ("getattr",) and len(target.args) == 2
            and chain(target.args[0]) == ("torch", "optim")
            and chain(target.args[1]) == (self_name, "config", "optimizer")
            and not target.keywords and not any(kw.arg is None for kw in call.keywords)):
        raise ValueError("The optimizer class is not selected by the supported config.optimizer lookup.")
    for next_call in straight_line_calls(train.body):
        if next_call.lineno > call.lineno and chain(next_call.func)[:3] == ("torch", "optim", "lr_scheduler"):
            trail.append(sources.site(path, next_call))
    return call


def hydra_recipe_lr(snapshot, reference):
    path = reference["entry_point"]
    sources = Sources(snapshot)
    candidate = {"value": None, "use": None, "definitions": [], "stage_trace": []}
    trail = candidate["stage_trace"]
    try:
        _, body = sources.read(path)
        entries = [call for node in body for call in straight_line_calls(main_body(node))
                   if isinstance(call.func, ast.Name) and not call.args and not call.keywords]
        entry = method(body, unique(entries, "Cannot resolve the Hydra entry call.").func.id)
        decorator = unique([node for node in entry.decorator_list if isinstance(node, ast.Call)
                            and chain(node.func) == ("hydra", "main")], "The selected script lacks a unique @hydra.main.")
        if len(entry.decorator_list) != 1 or not any(
            isinstance(node, ast.Import) and any(alias.name == "hydra" and not alias.asname for alias in node.names)
            for node in body
        ):
            raise ValueError("Cannot verify the Hydra decorator binding.")
        trail.append(sources.site(path, decorator))
        keywords = {kw.arg: kw.value for kw in decorator.keywords}
        subdir, filename = [getattr(keywords.get(key), "value", None) for key in ("config_path", "config_name")]
        if (decorator.args or set(keywords) - {"config_path", "config_name", "version_base"}
                or not isinstance(subdir, str) or not isinstance(filename, str)
                or not re.fullmatch(r"[\w-]+(?:/[\w-]+)*", subdir) or not COMPONENT.fullmatch(filename)):
            raise ValueError("Hydra config_path and config_name must be supported literals.")
        config_path = str(PurePosixPath(path).parent / subdir / f"{filename}.yaml")
        resolved = compose(snapshot, config_path, reference["arguments"], candidate)
        option = unique(formal(entry), "Cannot bind the Hydra configuration parameter.")
        worker, slot, routes = worker_route(body, entry, option)
        calls = {("OmegaConf", name) for name in ("resolve", "missing_keys", "save", "to_yaml", "to_container")}
        permitted = [arg for _, _, arg, _ in routes]
        permitted.extend(arg for node in ast.walk(entry) if isinstance(node, ast.Call) and chain(node.func) in calls
                         for arg in node.args if chain(arg) == (option,))
        if list(attribute_writes(entry, (option,), permitted)):
            raise ValueError("The entry point may change or dynamically pass Hydra optimizer settings.")
        trail.extend(sources.site(path, call) for _, _, _, call in routes)
        options = formal(worker)
        if len(options) <= slot:
            raise ValueError("Cannot bind the worker's Hydra configuration parameter.")
        # The worker may serialize its config for logging.
        worker_reads = [arg for node in ast.walk(worker) if isinstance(node, ast.Call) and chain(node.func) in calls
                        for arg in node.args if chain(arg) == (options[slot],)]
        trainer_path, classes, cfg_slot = trainer_use(sources, path, worker, options[slot], resolved, trail, worker_reads)
        optimizer_site(sources, trainer_path, classes, cfg_slot, candidate)
        if resolved.get("optimizer") not in OPTIMIZERS:
            raise ValueError("The selected optimizer name is not a supported literal.")
        candidate["optimizer"] = resolved["optimizer"]
        candidate["value"] = normalized(resolved["lr"])
    except (ValueError, SyntaxError) as error:
        return [candidate], str(error)
    return [candidate], None
