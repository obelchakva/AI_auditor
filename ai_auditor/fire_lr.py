"""Trace a documented Fire training subcommand and its base LR through warmup."""

import ast
import re
import shlex
from pathlib import PurePosixPath

from .experiment_lr import subject_key
from .flag_lr import Sources, chain
from .initial_lr import LAUNCH, literal, main_body, name_writes, normalized, straight_line_calls

README_POINTER = re.compile(r"\b(?:scripts|commands)\b.*?`([\w/-]+/?)(?:/)?`", re.I)
DOC_POINTER = re.compile(r"\b(?:commands?|launch)\b.*\b(?:comments?|docstrings?|top of the scripts?)\b", re.I)
SERVICE = {"bucket_name_prefix", "exp_name", "tpu_name", "log_dir"}
ENV = re.compile(r"\$(?:[A-Za-z_]\w*|\{[A-Za-z_]\w*\})")
SCALAR = re.compile(r"[\w./+-]+")


def one(items, message):
    if len(items) != 1:
        raise ValueError(message)
    return items[0]


def function(body, name):
    return one([item for item in body if isinstance(item, ast.FunctionDef)
                and item.name == name and not item.decorator_list],
               f"Cannot resolve one local function {name}.")


def signature(fn):
    if fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs:
        raise ValueError("Dynamic and positional-only training signatures are unsupported.")
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + fn.args.defaults + fn.args.kw_defaults
    return {arg.arg: value for arg, value in zip(args, defaults)}


def kwargs(call):
    values = {kw.arg: kw.value for kw in call.keywords}
    if None in values or len(values) != len(call.keywords):
        raise ValueError("The LR call has dynamic or duplicated keyword arguments.")
    return values


def parameter(call, name, expected):
    return chain(kwargs(call).get(name)) == (expected,)


def options(line):
    items = iter(shlex.split(line, comments=True))
    result = {}
    for token in items:
        flag, equals, value = token.partition("=")
        if not equals:
            value = next(items, "")
        flag_match = re.fullmatch(r"--([A-Za-z_]\w*)", flag)
        if not flag_match or flag_match[1] in result or not (SCALAR.fullmatch(value) or ENV.fullmatch(value)):
            raise ValueError("Unsupported or ambiguous arguments in the Fire training command.")
        result[flag_match[1]] = value
    return result


def discover(sources, dataset):
    snapshot = sources.snapshot
    recipes = []
    for readme in sorted(snapshot.files):
        if "/" in readme or PurePosixPath(readme).stem.casefold() != "readme":
            continue
        lines = snapshot.read(readme).splitlines()
        for index, line in enumerate(lines):
            directory = README_POINTER.search(line)
            if not directory:
                continue
            stop = min(len(lines), index + 4)
            pointers = [i for i in range(index, stop) if DOC_POINTER.search(lines[i])]
            if not pointers:
                continue
            prefix = directory[1].rstrip("/") + "/"
            for path in sorted(snapshot.files):
                if not path.startswith(prefix) or not path.endswith(".py"):
                    continue
                try:
                    source, body = sources.read(path)
                except SyntaxError:
                    continue
                if not body or not (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                                    and isinstance(body[0].value.value, str)):
                    continue
                doc = body[0]
                # Preserve exact source line numbers from the module docstring.
                source_lines = source.splitlines()
                for i in range(doc.lineno - 1, doc.end_lineno):
                    command = source_lines[i].strip()
                    launch = LAUNCH.match(command)
                    if not launch or launch["script"] != path:
                        continue
                    sub = re.fullmatch(r"\s+train(?:\s+(.*))?", command[launch.end():])
                    if not sub:
                        continue
                    try:
                        args = options(sub[1] or "")
                        fn = function(body, "train")
                        default = signature(fn).get("dataset")
                        selected = args.get("dataset", default.value if isinstance(default, ast.Constant) else "")
                    except ValueError:
                        args, selected = None, ""
                    doc_names_dataset = subject_key(dataset) in subject_key(doc.value.value.split("python", 1)[0])
                    if not (doc_names_dataset or isinstance(selected, str) and subject_key(selected) == subject_key(dataset)):
                        continue
                    recipe = {"path": path, "line_start": i + 1, "line_end": i + 1, "excerpt": command,
                              "entry_point": path, "subcommand": "train", "arguments": args,
                              "readme_pointer": {"path": readme, "line_start": index + 1,
                                                 "line_end": max(pointers) + 1,
                                                 "excerpt": "\n".join(lines[index:max(pointers) + 1])},
                              "dataset_context": {"path": path, "line_start": doc.lineno,
                                                  "line_end": i + 1,
                                                  "excerpt": "\n".join(source_lines[doc.lineno - 1:i]).strip()}}
                    if recipe not in recipes:
                        recipes.append(recipe)
    return recipes


def imported_module(sources, path, receiver):
    alias = one([node for node in sources.read(path)[1] if isinstance(node, ast.ImportFrom)
                 and (node.module or node.level)
                 and any((name.asname or name.name) == receiver for name in node.names)],
                f"Cannot resolve the module import for {receiver}.")
    name = next(name.name for name in alias.names if (name.asname or name.name) == receiver)
    parts = PurePosixPath(path).parts[:-1]
    if alias.level > len(parts):
        raise ValueError("The helper module import leaves the repository package.")
    package = parts[:len(parts) - alias.level + 1] if alias.level else ()
    module = ".".join((*package, *((alias.module,) if alias.module else ()), name))
    target = sources.module(module)
    if not target:
        raise ValueError("The imported helper module is absent or ambiguous on the pinned commit.")
    return target, sources.site(path, alias)


def qualified(call, member):
    ref = chain(call.func)
    return len(ref) == 2 and ref[-1] == member


def stable(body, *names):
    if any(name_writes(body, name) for name in names):
        raise ValueError("A tracked recipe parameter can change before reaching the optimizer.")


def dataset_registry(sources, path, selected):
    body = sources.read(path)[1]
    resolver = function(body, "get_dataset")
    names = list(signature(resolver))
    if not names:
        raise ValueError("The dataset provider lacks a dataset selector.")
    returns = [node for node in ast.walk(resolver) if isinstance(node, ast.Return)]
    returned = one(returns, "Cannot resolve one dataset provider return.")
    call = returned.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Name) and chain(call.func.slice) == (names[0],)):
        raise ValueError("The dataset provider does not dispatch by the training dataset parameter.")
    registry = call.func.value.id
    table = one([node.value for node in body if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and chain(node.targets[0]) == (registry,) and isinstance(node.value, ast.Dict)],
                "The training dataset registry is unresolved.")
    constructors = [value for key, value in zip(table.keys, table.values)
                    if isinstance(key, ast.Constant) and key.value == selected]
    constructor = one(constructors, "The selected dataset has no unique literal registry entry.")
    if (name_writes(body, registry) != 1 or None in table.keys or not isinstance(constructor, ast.Call)
            or chain(constructor.func) != ("functools", "partial")
            or not isinstance(kwargs(constructor).get("name"), ast.Constant)
            or kwargs(constructor)["name"].value != selected):
        raise ValueError("The registry does not confirm the identity of the selected dataset.")
    return [sources.site(path, node) for node in (constructor, returned)]


def warmup_definition(sources, path, fn):
    """Recognize the capped linear warmup used by the supported recipe."""
    if list(signature(fn)) != ["max_lr", "warmup", "global_step"] or len(fn.body) != 2:
        raise ValueError("The warmup function does not preserve a verifiable base LR.")
    zero, capped = fn.body
    if not (isinstance(zero, ast.If) and ast.unparse(zero.test) == "warmup == 0"
            and not zero.orelse and len(zero.body) == 1 and isinstance(zero.body[0], ast.Return)
            and chain(zero.body[0].value) == ("max_lr",) and isinstance(capped, ast.Return)):
        raise ValueError("The warmup function has an unsupported no-warmup branch.")
    form = ast.parse("max_lr * tf.minimum(tf.cast(global_step, tf.float32) / float(warmup), 1.0)", mode="eval").body
    if ast.dump(capped.value) != ast.dump(form):
        raise ValueError("The warmup expression is not the supported capped linear scaling of max_lr.")
    return [sources.site(path, zero), sources.site(path, capped)]


def trace_recipe(sources, recipe, dataset, trail):
    path = recipe["path"]
    source, body = sources.read(path)
    fn = function(body, recipe["subcommand"])
    fire = [call for node in body for call in straight_line_calls(main_body(node))
            if chain(call.func) == ("fire", "Fire")]
    dispatch = one(fire, "No unique guarded fire.Fire() dispatch.")
    if (dispatch.args or dispatch.keywords or sum(isinstance(node, ast.FunctionDef) and node.name == "train" for node in body) != 1
            or not any(isinstance(node, ast.Import) and any(alias.name == "fire" and not alias.asname for alias in node.names)
                       for node in body)):
        raise ValueError("Cannot bind the documented subcommand to the module's train function.")
    stable(body, "train", "fire")
    trail.append(sources.site(path, dispatch))

    params = signature(fn)
    args = recipe["arguments"]
    if args is None or args.keys() - params.keys() or any(value is None and name not in args for name, value in params.items()):
        raise ValueError("The documented command does not bind the Fire training signature.")
    if any(ENV.fullmatch(value) and name not in SERVICE for name, value in args.items()):
        raise ValueError("An environment substitution can change the experiment, optimizer or base LR.")
    if not {"dataset", "lr", "optimizer", "warmup"} <= params.keys():
        raise ValueError("The training function has no supported dataset and LR parameters.")
    for name in ("dataset", "optimizer"):
        value = params[name]
        effective = args.get(name, value.value if isinstance(value, ast.Constant) else None)
        target = subject_key(dataset) if name == "dataset" else "adam"
        if not isinstance(effective, str) or subject_key(effective) != target:
            raise ValueError(f"Cannot confirm the selected training {name}.")
        if value is not None and name not in args:
            trail.append(sources.site(path, value))
    stable(fn.body, "dataset", "lr", "optimizer", "warmup")
    base = normalized(args["lr"]) if "lr" in args else literal(source, params["lr"])
    steps = args.get("warmup", ast.get_source_segment(source, params["warmup"]) if params["warmup"] else "")
    if base is None or not re.fullmatch(r"\d+", steps or ""):
        raise ValueError("The documented base LR or warmup length is dynamic or unresolved.")
    definition = ([{key: recipe[key] for key in ("path", "line_start", "line_end", "excerpt")}]
                  if "lr" in args else [{
                      **sources.site(path, params["lr"]),
                      "parameter": "lr",
                      "excerpt": source.splitlines()[params["lr"].lineno - 1].strip(),
                  }])

    training = one([call for call in straight_line_calls(fn.body) if qualified(call, "run_training")],
                   "Cannot trace one training helper call.")
    if sum(isinstance(node, ast.Call) and qualified(node, "run_training") for node in ast.walk(fn)) != 1:
        raise ValueError("The recipe contains another possible training helper call.")
    if not all(parameter(training, name, name) for name in ("lr", "optimizer", "warmup")):
        raise ValueError("The training helper does not receive the resolved LR parameters directly.")
    dataset_calls = [call for call in straight_line_calls(fn.body) if qualified(call, "get_dataset")
                     and call.args and chain(call.args[0]) == ("dataset",)]
    dataset_call = one(dataset_calls, "Cannot trace the dataset parameter into the training input provider.")
    train_input = chain(kwargs(training).get("train_input_fn"))
    if len(train_input) != 2 or train_input[1] != "train_input_fn" or not any(
        isinstance(node, ast.Assign) and node.value is dataset_call and len(node.targets) == 1
        and chain(node.targets[0]) == train_input[:1] and node.lineno < training.lineno for node in fn.body
    ):
        raise ValueError("The training helper does not consume the selected dataset's training input.")
    dataset_path, dataset_link = imported_module(sources, path, chain(dataset_call.func)[0])
    dataset_default = params["dataset"]
    selected_dataset = args.get("dataset", getattr(dataset_default, "value", None))
    trail.extend([dataset_link, *dataset_registry(sources, dataset_path, selected_dataset)])
    helper_path, link = imported_module(sources, path, chain(training.func)[0])
    trail.extend([sources.site(path, dataset_call), sources.site(path, training), link])
    helper = function(sources.read(helper_path)[1], "run_training")
    if not {"lr", "warmup", "optimizer"} <= signature(helper).keys():
        raise ValueError("Cannot bind the base LR and optimizer to the training helper.")
    stable(helper.body, "lr", "optimizer", "warmup")
    estimator = one([node for node in helper.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                     and chain(node.value.func) == ("tf", "estimator", "tpu", "TPUEstimator")],
                    "Cannot resolve a unique training estimator.")
    callback_name = kwargs(estimator.value).get("model_fn")
    if not isinstance(callback_name, ast.Name):
        raise ValueError("Cannot resolve the estimator's training callback.")
    callback = function(helper.body, callback_name.id)
    launch = one([call for call in straight_line_calls(helper.body)
                  if chain(call.func) == (estimator.targets[0].id, "train")
                  and parameter(call, "input_fn", "train_input_fn") and call.lineno > estimator.lineno],
                 "The estimator does not train on the selected input provider.")
    if name_writes(helper.body, estimator.targets[0].id) != 1:
        raise ValueError("The training estimator is reassigned.")
    stable(helper.body, "train_input_fn", callback_name.id)
    trail.extend(sources.site(helper_path, node) for node in (estimator.value, launch))
    stable(callback.body, "lr", "optimizer", "warmup")
    warm = one([node for node in callback.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                and qualified(node.value, "get_warmed_up_lr")], "Cannot trace a unique warmup result.")
    make = one([call for call in straight_line_calls(callback.body) if qualified(call, "make_optimizer")],
               "Cannot trace a unique optimizer call in the training callback.")
    if (warm.lineno >= make.lineno or name_writes(callback.body, warm.targets[0].id) != 1
            or not parameter(warm.value, "max_lr", "lr") or not parameter(warm.value, "warmup", "warmup")
            or not parameter(make, "lr", warm.targets[0].id) or not parameter(make, "optimizer", "optimizer")
            or chain(warm.value.func)[0] != chain(make.func)[0]):
        raise ValueError("Cannot link the configured base LR through warmup into the optimizer.")
    if sum(isinstance(node, ast.Call) and qualified(node, "make_optimizer") for node in ast.walk(callback)) != 1:
        raise ValueError("The training callback contains another possible optimizer call.")
    step = chain(kwargs(warm.value).get("global_step"))
    if len(step) != 1 or name_writes(callback.body, step[0]) != 1:
        raise ValueError("Cannot resolve the runtime step used by warmup.")
    step_source = one([node for node in callback.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and chain(node.targets[0]) == step
                       and isinstance(node.value, ast.Call)
                       and chain(node.value.func) == ("tf", "train", "get_or_create_global_step")
                       and not node.value.args and not node.value.keywords and node.lineno < warm.lineno],
                      "Warmup does not use the supported TensorFlow global step.")
    trail.append(sources.site(helper_path, step_source))
    util, link = imported_module(sources, helper_path, chain(make.func)[0])
    trail.extend([sources.site(helper_path, warm), sources.site(helper_path, make), link])
    utility = sources.read(util)[1]
    warmup = warmup_definition(sources, util, function(utility, "get_warmed_up_lr"))
    factory = function(utility, "make_optimizer")
    stable(factory.body, "lr")
    if not {"lr", "optimizer"} <= signature(factory).keys():
        raise ValueError("Cannot bind the optimizer factory arguments.")
    branches = [node for node in factory.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "optimizer == 'adam'"]
    branch = one(branches, "Cannot resolve the Adam branch of the optimizer factory.")
    adam = one([call for call in straight_line_calls(branch.body) if chain(call.func) == ("tf", "train", "AdamOptimizer")
                and parameter(call, "learning_rate", "lr")], "Cannot trace the warmed rate to Adam.")
    trail.append(sources.site(util, branch.test))
    return {"value": base, "definitions": definition, "use": sources.site(util, adam), "stage_trace": trail,
            "rate_kind": "base_before_warmup", "warmup_steps": int(steps), "warmup_evidence": warmup,
            "base_lr_source": "command_override" if "lr" in args else "function_default"}


def transferred_fire_lr(snapshot, article):
    datasets = {item["experiment_evidence"]["dataset"] for item in article
                if item.get("scope_kind") == "dataset_transfer"}
    if len(datasets) != 1:
        return [], [], None, None
    dataset = next(iter(datasets))
    sources = Sources(snapshot)
    recipes = discover(sources, dataset)
    if not recipes:
        return [], recipes, "recipe_not_found", "No documented Fire training command matches the appendix dataset."
    if len(recipes) != 1:
        return [], recipes, "recipe_ambiguous", "Multiple documented Fire training commands match the appendix dataset."
    trail = []
    try:
        return [trace_recipe(sources, recipes[0], dataset, trail)], recipes, None, None
    except (ValueError, SyntaxError) as error:
        recipes[0]["stage_trace"] = trail
        return [], recipes, "code_lr_unresolved", str(error)
