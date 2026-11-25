"""Static checks for YAML-backed optimizer parameter groups."""

import ast
import re
import shlex
import unicodedata
from pathlib import PurePosixPath

import pymupdf
import yaml

from .initial_lr import (
    NUMBER, assignments, compact, evidence, is_optimizer, main_body,
    module_paths, name_writes, normalized, straight_line_calls,
)

PAIR = re.compile(
    rf"\blearning rates (?:are|were) initially set to ({NUMBER}) and ({NUMBER})"
    r" for (?:the )?([^,]+?) and (?:the )?([^,]+?), respectively\b", re.I,
)
SECTION = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{1,80}$")
GROUPS = {
    "pretrained weights": {"pretrained layers", "pretrained weights", "pretrained backbone weights"},
    "pretrained biases": {"pretrained biases", "pretrained backbone biases"},
    "scratch weights": {"classifier", "classifier weights", "scratch weights"},
    "scratch biases": {"classifier biases", "scratch biases"},
}


def words(text):
    return compact(unicodedata.normalize("NFKC", text).replace("-", "")).casefold()


def paper_blocks(page):
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        parts = []
        heading_words = []
        for line in block["lines"]:
            for span in line["spans"]:
                text = unicodedata.normalize("NFKC", span["text"]).strip()
                if not text:
                    continue
                bold = span["flags"] & 16 or re.search(r"bold|medi|demi", span["font"], re.I)
                if bold and not SECTION.fullmatch(text):
                    heading_words.append(text)
                    if text.endswith("."):
                        title = " ".join(heading_words)
                        parts.append((len(title.split()) <= 9, title))
                        heading_words.clear()
                    continue
                if heading_words:
                    parts.append((False, " ".join(heading_words)))
                    heading_words.clear()
                parts.append((bool(SECTION.fullmatch(text)), text))
        if heading_words:
            parts.append((False, " ".join(heading_words)))
        raw = compact(" ".join(text for _, text in parts))
        if re.match(r"^(?:Table|Figure)\s+\d+[:.]", raw, re.I):
            continue
        if any(heading or len(text.split()) >= 5 for heading, text in parts):
            yield parts


def grouped_paper(pdf, stage, group):
    candidates = []
    previous = None
    with pymupdf.open(pdf) as document:
        if not document.is_pdf or document.needs_pass:
            raise ValueError("article must be a readable, unencrypted PDF")
        for page in document:
            anchor, carried, section = previous, previous is not None, None
            previous = None
            for block in paper_blocks(page):
                prose = []
                for is_heading, text in [*block, (True, "")]:
                    if not is_heading:
                        prose.append(text)
                        continue
                    paragraph = compact(" ".join(prose))
                    prose = []
                    if anchor and paragraph:
                        # Require an immediate prose continuation of an unfinished sentence.
                        if not carried or re.match(r"^(?:[\w-]+ optimizer\b|learning rates\b)", paragraph, re.I):
                            for match in PAIR.finditer(paragraph):
                                for raw, label in zip(match.group(1, 2), match.group(3, 4)):
                                    value = normalized(raw)
                                    if words(label) in GROUPS.get(group, ()) and value is not None:
                                        candidates.append({
                                            "page": page.number + 1, "value": value, "group_label": words(label),
                                            "excerpt": match[0], "stage_context": anchor,
                                        })
                        if carried:
                            anchor = None
                        elif re.search(r"\b(?:using|with|via|by) (?:the|an|a)$", paragraph, re.I):
                            previous = {**anchor, "continuation": paragraph[-220:]}
                        else:
                            previous = None
                        carried = False
                    if text:
                        if SECTION.fullmatch(text):
                            section = text
                        anchor = (
                            {"page": page.number + 1, "excerpt": text, "section": section}
                            if words(text.rstrip(".")) == words(stage) else None
                        )
                        carried = False
                        previous = None
    return candidates


def attribute_path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        root = attribute_path(node.value)
        return (*root, node.attr) if root else ()
    return ()


def yaml_launch(reference):
    line = reference["excerpt"]
    match = re.search(r"\bpython(?:3(?:\.\d+)?)?\s+(?:\./)?[\w./-]+\.py\s+(.*)", line)
    if not match:
        return None
    try:
        args = shlex.split(match[1].strip(chr(96) + "*").split(chr(96), 1)[0], comments=True)
    except ValueError:
        return None
    if len(args) % 2 or not all(re.fullmatch(r"--[\w-]+", flag) for flag in args[::2]):
        return None
    options = dict(zip(args[::2], args[1::2]))
    files = [(flag, path) for flag, path in options.items()
             if re.fullmatch(r"(?:\./)?[\w./-]+\.ya?ml", path, re.I)]
    if len(files) != 1 or len(options) != len(args) // 2:
        return None
    flag, location = files[0]
    path = PurePosixPath(location)
    if path.is_absolute() or ".." in path.parts:
        return None
    return flag, str(path), set(options) - {flag}


def yaml_scalar(path, source, keys):
    # Compose reads nodes and source marks without calling YAML constructors.
    if any(isinstance(token, (yaml.tokens.TagToken, yaml.tokens.AliasToken))
           for token in yaml.scan(source, Loader=yaml.SafeLoader)):
        raise ValueError("The selected YAML contains explicit tags or aliases.")
    node = yaml.compose(source, Loader=yaml.SafeLoader)
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            return None
        matches = [(k, v) for k, v in node.value if isinstance(k, yaml.ScalarNode) and k.value == key]
        if len(matches) != 1:
            return None
        key_node, node = matches[0]
    if not isinstance(node, yaml.ScalarNode) or node.style is not None or not re.fullmatch(NUMBER, node.value):
        return None
    start, end = key_node.start_mark.line, node.end_mark.line
    return normalized(node.value), {
        "path": path, "line_start": start + 1, "line_end": end + 1,
        "excerpt": "\n".join(source.splitlines()[start:end + 1]).strip(),
    }


def parser_options(function):
    last = function.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call):
        return {}
    result = last.value
    parser = attribute_path(result.func)
    if len(parser) != 2 or parser[1] != "parse_args" or result.args or result.keywords:
        return {}
    return {
        call.args[0].value: call
        for call in straight_line_calls(function.body)
        if attribute_path(call.func) == (parser[0], "add_argument")
        and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
        and not any(k.arg in {"dest", "action", None} for k in call.keywords)
        and all(k.arg != "type" or isinstance(k.value, ast.Name) and k.value.id == "str"
                for k in call.keywords)
    }


def device_only(guards, args, flag):
    attr = (args, flag[2:].replace("-", "_"))
    uses = [n for g in guards for n in ast.walk(g)
            if isinstance(n, ast.Attribute) and attribute_path(n) == attr]
    return len(uses) == 1 and any(
        isinstance(g, ast.Assign) and g.value is uses[0] and len(g.targets) == 1
        and isinstance(g.targets[0], ast.Subscript) and attribute_path(g.targets[0].value) == ("os", "environ")
        and isinstance(g.targets[0].slice, ast.Constant)
        and g.targets[0].slice.value == "CUDA_VISIBLE_DEVICES" for g in guards
    )


def config_route(body, flag, other_flags):
    functions = {node.name: node for node in body if isinstance(node, ast.FunctionDef)}
    guards = [node for statement in body for node in main_body(statement)]
    routes = []
    for statement in guards:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, ast.Call)):
            continue
        fn = functions.get(getattr(statement.value.func, "id", ""))
        if not fn or statement.value.args or statement.value.keywords:
            continue
        declarations = parser_options(fn)
        args = statement.targets[0].id
        if flag not in declarations or not all(device_only(guards, args, f) for f in other_flags):
            continue
        for load in straight_line_calls(guards):
            receiver = attribute_path(load.func)
            if (len(receiver) != 2 or receiver[1] != "merge_from_file"
                    or len(load.args) != 1 or load.keywords
                    or attribute_path(load.args[0]) != (args, flag[2:].replace("-", "_"))
                    or load.lineno <= statement.lineno):
                continue
            config = receiver[0]
            for call in straight_line_calls(guards):
                train = functions.get(getattr(call.func, "id", ""))
                if (train and call.lineno > load.lineno and len(call.args) == 1 and not call.keywords
                        and attribute_path(call.args[0]) == (config,) and len(train.args.args) == 1
                        and name_writes(guards, config) == 1):
                    changes = [n for g in guards if load.lineno < g.lineno < call.lineno
                               for n in ast.walk(g) if isinstance(n, (ast.Call, ast.Attribute))
                               and attribute_path(n.func if isinstance(n, ast.Call) else n)[:1] == (config,)
                               and attribute_path(n.func if isinstance(n, ast.Call) else n) != (config, "freeze")]
                    if not changes:
                        routes.append((train, [declarations[flag], statement, load, call]))
    return routes


def index_of(node, name):
    if (isinstance(node, ast.Subscript) and attribute_path(node.value) == (name,)
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int and node.slice.value >= 0):
        return node.slice.value
    return None


def parameter_origins(snapshot, path, source, body, train, param_node):
    """Prove group provenance from parameter collection and a loaded backbone."""
    getter = param_node.value
    method = attribute_path(getter.func) if isinstance(getter, ast.Call) else ()
    if len(method) != 2 or getter.args or getter.keywords:
        return {}
    model, method_name = method
    objects = assignments(train.body, model)
    if len(objects) != 1 or name_writes(train.body, model) != 1:
        return {}
    ctor = objects[0].value
    if (isinstance(ctor, ast.Call) and isinstance(ctor.func, ast.Attribute)
            and ctor.func.attr in {"cuda", "to"}):
        ctor = ctor.func.value
    if not isinstance(ctor, ast.Call) or not isinstance(ctor.func, ast.Name):
        return {}
    imports = [(n.module, item.name) for n in body if isinstance(n, ast.ImportFrom) and not n.level and n.module
               for item in n.names if (item.asname or item.name) == ctor.func.id]
    if len(imports) != 1:
        return {}
    locations = module_paths(snapshot, imports[0][0], entry=False)
    if len(locations) != 1:
        return {}
    model_path = locations[0]
    model_source = snapshot.read(model_path)
    classes = [n for n in ast.parse(model_source).body if isinstance(n, ast.ClassDef) and n.name == imports[0][1]]
    if len(classes) != 1:
        return {}
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    collect, init = methods.get(method_name), methods.get("__init__")
    if not collect or not init or not collect.args.args or not init.args.args:
        return {}
    returns = [n for n in ast.walk(collect) if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Name):
        return {}
    name = returns[0].value.id
    start = assignments(collect.body, name)
    if (len(start) != 1 or name_writes(collect.body, name) != 1
            or not isinstance(start[0].value, (ast.List, ast.Tuple))
            or not all(isinstance(item, ast.List) and not item.elts for item in start[0].value.elts)):
        return {}
    load_calls = [c for c in straight_line_calls(train.body)
                  if len(attribute_path(c.func)) == 3 and attribute_path(c.func)[0] == model
                  and attribute_path(c.func)[2] == "load_state_dict" and c.args
                  and objects[0].lineno < c.lineno < param_node.lineno]
    if len(load_calls) != 1:
        return {}
    receiver = collect.args.args[0].arg
    init_receiver = init.args.args[0].arg
    loaded = attribute_path(load_calls[0].func)[1]
    constructed_layers = {
        attribute_path(n.targets[0])[1]
        for n in init.body if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.value, ast.Call) and len(attribute_path(n.targets[0])) == 2
        and attribute_path(n.targets[0])[0] == init_receiver
    }
    if loaded not in constructed_layers:
        return {}
    origins = {}
    recognized = set()
    for loop in collect.body:
        if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
                and isinstance(loop.iter, ast.Call) and attribute_path(loop.iter.func) == (receiver, "modules")):
            continue
        for branch in ast.walk(loop):
            if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare):
                continue
            test = branch.test
            if (len(test.ops) != 1 or not isinstance(test.ops[0], ast.In)
                    or attribute_path(test.left) != (loop.target.id,)):
                continue
            scratch = attribute_path(test.comparators[0])
            if len(scratch) != 2 or scratch[0] != receiver or "scratch" not in scratch[1].lower():
                continue
            declared = [n for n in init.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                        and attribute_path(n.targets[0]) == (init_receiver, scratch[1])
                        and isinstance(n.value, (ast.List, ast.Tuple)) and n.value.elts
                        and all(len(attribute_path(item)) == 2 and attribute_path(item)[0] == init_receiver
                                and attribute_path(item)[1] != loaded for item in n.value.elts)]
            if len(declared) != 1:
                continue
            scratch_layers = {attribute_path(item)[1] for item in declared[0].value.elts}
            if constructed_layers - scratch_layers != {loaded}:
                continue
            for label, branch_body in (("scratch", branch.body), ("pretrained", branch.orelse)):
                for call in straight_line_calls(branch_body):
                    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "append"
                            and len(call.args) == 1 and not call.keywords):
                        continue
                    index = index_of(call.func.value, name)
                    attribute = attribute_path(call.args[0])
                    if (index is None or index >= len(start[0].value.elts) or len(attribute) != 2
                            or attribute[0] != loop.target.id or attribute[1] not in {"weight", "bias"}):
                        continue
                    recognized.add(call)
                    plural = "weights" if attribute[1] == "weight" else "biases"
                    trail = [evidence(model_path, model_source, declared[0]),
                             evidence(model_path, model_source, test), evidence(model_path, model_source, call)]
                    origins.setdefault(index, {}).setdefault(f"{label} {plural}", []).extend(trail)
    # A conflicting or unclassified writer makes that entire list unsafe to use.
    for n in ast.walk(collect):
        index = index_of(n, name)
        if index is not None and isinstance(n.ctx, (ast.Store, ast.Del)):
            origins.pop(index, None)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n not in recognized and index_of(n.func.value, name) is not None):
            origins.pop(index_of(n.func.value, name), None)
    return {
        index: (next(iter(labels)), [evidence(path, source, load_calls[0]), *next(iter(labels.values()))])
        for index, labels in origins.items() if len(labels) == 1
    }


def group_dict(node):
    if not isinstance(node, ast.Dict) or not all(isinstance(k, ast.Constant) for k in node.keys):
        return {}
    return dict(zip((k.value for k in node.keys), node.values)) if len({k.value for k in node.keys}) == len(node.keys) else {}


def grouped_code(snapshot, reference, group):
    launch = yaml_launch(reference)
    if not launch:
        raise ValueError("The documented launch does not select a single literal YAML file.")
    flag, yaml_path, other_flags = launch
    if yaml_path not in snapshot.files:
        raise ValueError("The YAML selected in README is absent from the selected commit.")
    path = reference["entry_point"]
    source = snapshot.read(path)
    body = ast.parse(source).body
    routes = config_route(body, flag, other_flags)
    if len(routes) != 1:
        raise ValueError("Could not uniquely trace the YAML option to a training function.")
    train, route = routes[0]
    config = train.args.args[0].arg
    if name_writes(train.body, config) or any(
        isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del))
        and attribute_path(n.value)[:1] == (config,) for n in ast.walk(train)
    ):
        raise ValueError("The loaded configuration can be overridden during training.")
    direct = [call for call in straight_line_calls(train.body) if is_optimizer(call)]
    all_optimizers = [n for n in ast.walk(train) if isinstance(n, ast.Call) and is_optimizer(n)]
    if len(direct) != 1 or len(all_optimizers) != 1 or not direct[0].args or not isinstance(
        direct[0].args[0], (ast.List, ast.Tuple)
    ):
        raise ValueError("Could not isolate one optimizer with explicit parameter groups.")
    optimizer = direct[0]
    candidates = []
    origins_by_list = {}
    for spec in optimizer.args[0].elts:
        entries = group_dict(spec)
        params = entries.get("params")
        lr = entries.get("lr")
        if not isinstance(params, ast.Subscript) or not isinstance(params.value, ast.Name):
            raise ValueError("An optimizer parameter group has an unrecognized parameter source.")
        index = index_of(params, params.value.id)
        definitions = assignments(train.body, params.value.id)
        if index is None or len(definitions) != 1 or name_writes(train.body, params.value.id) != 1:
            raise ValueError("An optimizer parameter group has an ambiguous parameter source.")
        param_node = definitions[0]
        if param_node.lineno >= optimizer.lineno:
            raise ValueError("The parameter list is defined after the optimizer.")
        if param_node not in origins_by_list:
            origins_by_list[param_node] = parameter_origins(snapshot, path, source, body, train, param_node)
        origins = origins_by_list[param_node]
        if index not in origins:
            raise ValueError("Could not prove the membership of all optimizer parameter groups.")
        label, origin = origins[index]
        if label != group:
            continue
        # This slice resolves literal config reads and a single local alias.
        alias = assignments(train.body, lr.id) if isinstance(lr, ast.Name) else []
        if isinstance(lr, ast.Name) and (
            len(alias) != 1 or name_writes(train.body, lr.id) != 1 or alias[0].lineno >= optimizer.lineno
        ):
            raise ValueError("The LR alias for the selected group is ambiguous.")
        keys = attribute_path(alias[0].value if alias else lr)
        if len(keys) < 2 or keys[0] != config:
            raise ValueError("The selected group's LR is not a direct YAML value or a simple alias.")
        scalar = yaml_scalar(yaml_path, snapshot.read(yaml_path), keys[1:])
        if not scalar or scalar[0] is None:
            raise ValueError("The selected YAML does not have a unique numeric value at " + ".".join(keys[1:]) + ".")
        value, location = scalar
        candidates.append({
            "group_label": label, "group_index": index, "value": value,
            "use": evidence(path, source, spec), "definitions": [location, *(
                [evidence(path, source, alias[0])] if alias else []
            )],
            "stage_trace": [*(evidence(path, source, n) for n in route),
                            evidence(path, source, param_node), *origin],
        })
    if len(candidates) != 1:
        raise ValueError("Could not resolve a unique optimizer group for the requested selector.")
    return candidates
