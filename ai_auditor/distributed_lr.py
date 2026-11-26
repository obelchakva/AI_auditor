"""Inspect one documented distributed Transformers training recipe without executing it."""

import ast
import re
import shlex
from pathlib import PurePosixPath

from .experiment_lr import subject_key
from .flag_lr import Sources, chain
from .initial_lr import NUMBER, normalized
from .shell_lr import PYTHON, line_site, relative


REQUIRED = {"model_name_or_path", "task_name", "do_train", "max_seq_length", "per_device_train_batch_size",
            "learning_rate", "num_train_epochs", "warmup_ratio", "apply_lora", "lora_r", "lora_alpha"}
EXTRA = {"do_eval", "output_dir", "logging_steps", "logging_dir", "evaluation_strategy", "save_strategy",
         "seed", "weight_decay", "gradient_accumulation_steps"}
VARIABLE = re.compile(r"\$([A-Za-z_]\w*)")


def only(items, message):
    if len(items) != 1:
        raise ValueError(message)
    return items[0]


def shell_launch(snapshot, path):
    """Only literal exports and one Python command joined with backslashes."""
    exports, command, sites = {}, [], []
    pending, finished = False, False
    for line, raw in enumerate(snapshot.read(path).splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            if pending:
                raise ValueError("Unsupported comment or blank line in the continued launch.")
            continue
        try:
            tokens = shlex.split(text.removesuffix("\\"), comments=True)
        except ValueError as error:
            raise ValueError("The shell recipe contains an unresolved command.") from error
        if not pending and not finished and tokens and tokens[0] == "export":
            binding = re.fullmatch(r"([A-Za-z_]\w*)=([\w./:+-]+)", tokens[1]) if len(tokens) == 2 else None
            if not binding or binding[1] in exports:
                raise ValueError("The shell recipe contains a dynamic or repeated export.")
            exports[binding[1]] = binding[2]
            sites.append(line_site(path, line, raw))
            continue
        if finished or not tokens or not pending and not PYTHON.fullmatch(tokens[0]):
            raise ValueError("The selected script contains an additional or unsupported shell command.")
        command.extend(tokens)
        sites.append(line_site(path, line, raw))
        pending = text.endswith("\\")
        finished = not pending
    if pending or command[1:3] != ["-m", "torch.distributed.launch"] or len(command) < 6:
        raise ValueError("The selected script does not contain a supported distributed Python command.")
    launcher = re.fullmatch(r"--nproc_per_node=(\d+|\$[A-Za-z_]\w*)", command[3])
    if launcher is None:
        raise ValueError("The number of training processes is unresolved.")
    variable = VARIABLE.fullmatch(launcher[1])
    count = exports.get(variable[1], "") if variable else launcher[1]
    if not re.fullmatch(r"[1-9]\d{0,5}", count):
        raise ValueError("The number of training processes is not a positive literal.")
    if not re.fullmatch(r"[\w./-]+\.py", command[4]):
        raise ValueError("The Python entry script is not literal.")
    argv = iter(command[5:])
    arguments = {}
    for token in argv:
        option, equals, value = token.partition("=")
        name = option.removeprefix("--")
        if not option.startswith("--") or name not in REQUIRED | EXTRA or name in arguments:
            raise ValueError("The recipe contains an unsupported or repeated training argument.")
        if name in {"apply_lora", "do_train", "do_eval"}:
            if equals:
                raise ValueError("Unsupported boolean training option.")
            arguments[name] = True
            continue
        if not equals:
            value = next(argv, "")
        if name in {"output_dir", "logging_dir"}:
            value = VARIABLE.sub(lambda match: exports.get(match[1], match[0]), value)
        if not re.fullmatch(r"[\w./+-]+", value):
            raise ValueError(f"The recipe's {name} is dynamic or unresolved.")
        arguments[name] = value
    if not REQUIRED <= arguments.keys() or not re.fullmatch(NUMBER, arguments["learning_rate"]):
        raise ValueError("The recipe lacks the explicit experiment settings or a numeric learning rate.")
    return relative(PurePosixPath(path).parent, command[4]), int(count), arguments, sites


def commit_readmes(snapshot, path):
    folder = str(PurePosixPath(path).parent)
    root, nested = [], []
    for file in snapshot.files:
        if PurePosixPath(file).stem.casefold() != "readme":
            continue
        parent = str(PurePosixPath(file).parent)
        if parent not in {".", folder}:
            continue
        for line, raw in enumerate(snapshot.read(file).splitlines(), 1):
            if parent == "." and folder in raw and re.search(r"\breproduc\w+|\bresults\b", raw, re.I):
                root.append(line_site(file, line, raw))
            if parent == folder and ("shell scripts" in raw.casefold() or PurePosixPath(path).name in raw):
                nested.append(line_site(file, line, raw))
    if not root or not nested:
        raise ValueError("Cannot link the selected script to the root and experiment README.")
    return sorted(root + nested, key=lambda item: (item["path"], item["line_start"]))


def resolve_member(sources, path, name, kind):
    return only([node for node in sources.read(path)[1] if isinstance(node, kind) and node.name == name],
                f"Cannot resolve {name} at {path}.")


def method(cls, name):
    return only([node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name],
                f"Cannot resolve framework method {name}.")


def field(cls, name):
    return only([node for node in cls.body if isinstance(node, ast.AnnAssign)
                 and chain(node.target) == (name,)], f"Cannot resolve training field {name}.")


def default(node):
    if isinstance(node.value, ast.Call) and chain(node.value.func) == ("field",):
        return next((kw.value.value for kw in node.value.keywords
                     if kw.arg == "default" and isinstance(kw.value, ast.Constant)), None)
    return getattr(node.value, "value", None)


def matching(nodes, text, message):
    return only([node for node in nodes if ast.unparse(node) == text], message)


def framework_trace(sources, entry, root, arguments, count):
    if str(PurePosixPath(entry).with_suffix(".args")) in sources.snapshot.files:
        raise ValueError("A tracked .args file can supplement the shell arguments.")
    src = PurePosixPath(root, "src", "transformers")
    args_path, trainer_path, parser_path = (str(src / name) for name in
                                           ("training_args.py", "trainer.py", "hf_argparser.py"))
    if not all(path in sources.snapshot.files for path in (entry, args_path, trainer_path, parser_path)):
        raise ValueError("The selected recipe lacks the pinned parser, training arguments or Trainer.")
    fn = resolve_member(sources, entry, "main", ast.FunctionDef)
    parse = matching(ast.walk(fn), "model_args, data_args, training_args = parser.parse_args_into_dataclasses()",
                     "Cannot bind command arguments to training dataclasses.")
    bind = matching(fn.body, "parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))",
                    "Cannot bind the supported argument parser.")
    trainer_call = only([node for node in ast.walk(fn) if isinstance(node, ast.Call) and chain(node.func) == ("Trainer",)
                         and any(kw.arg == "args" and chain(kw.value) == ("training_args",) for kw in node.keywords)],
                        "Cannot bind the parsed training arguments to Trainer.")
    if trainer_call.lineno <= parse.lineno or any(
        isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))
        and chain(n) == ("training_args", "learning_rate") for n in ast.walk(fn)
    ):
        raise ValueError("The parsed LR can change before entering Trainer.")
    if not any(isinstance(node, ast.Call) and chain(node.func) == ("trainer", "train") for node in ast.walk(fn)):
        raise ValueError("The documented Trainer never starts training.")
    parser = resolve_member(sources, parser_path, "HfArgumentParser", ast.ClassDef)
    type_binding = matching(ast.walk(method(parser, "_add_dataclass_arguments")), "kwargs['type'] = field.type",
                            "Cannot verify CLI type conversion.")
    argv_parse = method(parser, "parse_args_into_dataclasses")
    parsed = matching(ast.walk(argv_parse), "namespace, remaining_args = self.parse_known_args(args=args)",
                      "Cannot verify parsing of the CLI arguments.")
    converted = matching(ast.walk(argv_parse), "obj = dtype(**inputs)", "Cannot verify dataclass construction.")
    settings = resolve_member(sources, args_path, "TrainingArguments", ast.ClassDef)
    lr_field = field(settings, "learning_rate")
    if chain(lr_field.annotation) != ("float",) or default(field(settings, "adafactor")) is not False:
        raise ValueError("The pinned training arguments do not select the supported numeric AdamW route.")
    ga_field = field(settings, "gradient_accumulation_steps")
    ga = str(arguments.get("gradient_accumulation_steps", default(ga_field)))
    batch = arguments["per_device_train_batch_size"]
    if not all(re.fullmatch(r"[1-9]\d{0,5}", value) for value in (ga, batch)):
        raise ValueError("Cannot resolve positive batch size and gradient accumulation.")
    per_gpu = field(settings, "per_gpu_train_batch_size")
    if default(per_gpu) is not None:
        raise ValueError("A deprecated batch-size setting may override the per-device value.")
    local = method(settings, "train_batch_size")
    per_device = matching(ast.walk(local),
                          "per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size",
                          "Cannot verify the per-device batch setting.")
    setup = method(settings, "_setup_devices")
    one_gpu = [n for n in ast.walk(setup)
               if isinstance(n, ast.Assign) and ast.unparse(n) == "self._n_gpu = 1"]
    ddp = only([node for node in ast.walk(method(settings, "_setup_devices")) if isinstance(node, ast.If)
                and any(isinstance(call, ast.Call) and chain(call.func) == ("torch", "distributed", "init_process_group")
                        for stmt in node.orelse for call in ast.walk(stmt))
                and any(n in node.orelse for n in one_gpu)], "Cannot verify one device per distributed process.")
    local_size = matching(ast.walk(local), "train_batch_size = per_device_batch_size * max(1, self.n_gpu)",
                          "Cannot verify the per-process batch size calculation.")

    trainer = resolve_member(sources, trainer_path, "Trainer", ast.ClassDef)
    init = matching(method(trainer, "__init__").body, "self.args = args", "Cannot trace arguments into Trainer.")
    optimizer = method(trainer, "create_optimizer")
    selected = matching(ast.walk(optimizer), "optimizer_cls = AdamW", "Cannot locate the AdamW branch.")
    forwarded = matching(optimizer.body, "optimizer_kwargs['lr'] = self.args.learning_rate",
                         "Cannot verify the LR passed to the optimizer.")
    use = matching(optimizer.body, "return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)",
                   "Cannot resolve the optimizer construction.")
    training = method(trainer, "train")
    size = matching(ast.walk(training),
                    "total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size",
                    "Cannot verify effective batch-size calculation.")
    world = matching(ast.walk(training), "world_size = dist.get_world_size()",
                     "Cannot trace the distributed world size.")
    warmup = method(trainer, "create_scheduler")
    warmup_formula = only([node for node in ast.walk(warmup) if isinstance(node, ast.Call)
                           and ast.unparse(node) == "math.ceil(num_training_steps * self.args.warmup_ratio)"],
                          "Cannot verify schedule warmup separately.")
    if default(field(settings, "warmup_steps")) != 0:
        raise ValueError("The warmup ratio can be superseded by an unresolved step count.")
    launch_config = only([node for node in ast.walk(fn) if isinstance(node, ast.Call)
                          and chain(node.func) == ("AutoConfig", "from_pretrained")
                          and all(any(kw.arg == key and chain(kw.value) == ("model_args", key) for kw in node.keywords)
                                  for key in ("apply_lora", "lora_r", "lora_alpha"))],
                         "Cannot verify the enabled adapter, rank and alpha passed to model configuration.")
    family = arguments["model_name_or_path"].partition("-")[0]
    model_path = str(src / "models" / family / f"modeling_{family}.py")
    if model_path not in sources.snapshot.files:
        raise ValueError("The pinned adapter implementation for the model family is unavailable.")
    model_nodes = sources.read(model_path)[1]
    projections = []
    for projection in ("query", "value"):
        matches = [node for item in model_nodes for node in ast.walk(item) if isinstance(node, ast.If)
                   and chain(node.test) == ("config", "apply_lora")
                   for node in node.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and chain(node.targets[0]) == ("self", projection) and isinstance(node.value, ast.Call)
                   and chain(node.value.func) == ("lora", "Linear")
                   and any(chain(arg) == ("config", "lora_r") for arg in node.value.args)
                   and any(kw.arg == "lora_alpha" and chain(kw.value) == ("config", "lora_alpha")
                           for kw in node.value.keywords)]
        projections.append(only(matches, f"Cannot confirm the adapter rank/alpha for {projection}."))
    gpu = next(n for n in one_gpu if n in ddp.orelse)
    return {
        "use": sources.site(trainer_path, use),
        "stage_trace": [*(sources.site(entry, n) for n in (bind, parse, launch_config, trainer_call)),
                        *(sources.site(parser_path, n) for n in (type_binding, parsed, converted)),
                        sources.site(args_path, lr_field),
                        *(sources.site(trainer_path, n) for n in (init, selected, forwarded)),
                        *(sources.site(model_path, n) for n in projections)],
        "warmup": {"ratio": arguments["warmup_ratio"], "proof": sources.site(trainer_path, warmup_formula)},
        "batch": {"per_device": int(batch), "processes": count, "devices_per_process": 1,
                  "gradient_accumulation": int(ga), "effective": int(batch) * count * int(ga),
                  "evidence": [*(sources.site(args_path, n) for n in (ga_field, per_device, gpu, local_size)),
                               sources.site(trainer_path, world), sources.site(trainer_path, size)]},
    }


def comparison_issue(article, recipe):
    if not article:
        return "The table does not verify a comparison group for the published recipe."
    chosen, args = article[0], recipe["arguments"]
    if (subject_key(args["task_name"]) != subject_key(chosen["dataset"])
            or not subject_key(chosen["model_method"]).startswith(subject_key(args["model_name_or_path"]))):
        return "The selected table model or dataset does not identify this recipe."
    settings = chosen.get("group_settings", {})
    batch = settings.get("Batch Size", {}).get("value")
    checks = {}
    for label, option in (("# Epochs", "num_train_epochs"), ("LoRA α", "lora_alpha"),
                          ("Max Seq. Len.", "max_seq_length"), ("LoRA Config.", "lora_r")):
        paper_setting = next((item["value"] for key, item in settings.items()
                              if subject_key(key) == subject_key(label)), None)
        actual = args.get(option)
        if option == "lora_r" and paper_setting:
            rank = re.fullmatch(r"r\s*q\s*=\s*r\s*v\s*=\s*(\d+)", paper_setting, re.I)
            agrees = bool(rank and rank[1] == actual)
        else:
            agrees = paper_setting is not None and paper_setting == actual
        checks[option] = {"paper": paper_setting, "recipe": actual, "agrees": agrees}
    recipe["comparison"] = {"table_label": chosen["model_method"], "table_batch": batch,
                            "per_device_batch": recipe["batch"]["per_device"],
                            "effective_batch": recipe["batch"]["effective"],
                            "table_batch_scope": "unspecified", "variant_confirmed": False,
                            "settings": checks}
    return ("The published recipe's base LR is verified. The table does not establish whether its "
            "batch size means per-device or effective batch size, or whether the command belongs "
            "to the selected table variant.")


def table_shell_audit(snapshot, path, article):
    links, code = [], []
    try:
        if not path.endswith(".sh") or path not in snapshot.files or relative(".", path) != path:
            raise ValueError("The selected shell script is not a pinned relative path.")
        readmes = commit_readmes(snapshot, path)
        links.extend(readmes)
        entry, count, args, shell = shell_launch(snapshot, path)
        links.extend(shell)
        base = normalized(args["learning_rate"])
        if base is None:
            raise ValueError("The shell learning rate is invalid.")
        traced = framework_trace(Sources(snapshot), entry, str(PurePosixPath(path).parent), args, count)
        code.append({**traced, "value": base, "rate_kind": "base_before_warmup", "arguments": args,
                     "definitions": [item for item in shell if "--learning_rate" in item["excerpt"]],
                     "scope": "Published launch on one node with fresh output and no local .args file."})
    except (ValueError, SyntaxError) as error:
        return code, links, str(error)
    return code, links, comparison_issue(article, code[0])
