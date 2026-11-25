"""Static trace for documented shell dispatch into an argparse-backed SGD recipe."""

import ast
import re
import shlex
from pathlib import PurePosixPath

from .flag_lr import Sources, chain
from .initial_lr import NUMBER, describes_stage, name_writes, normalized, stage_pattern

FUNCTION = re.compile(r"([A-Za-z_]\w*)\s*\(\s*\)\s*\{")
CHOICES = re.compile(r"<([\w-]+(?:\|[\w-]+)+)>")
PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?")
FIELDS = {"dataset": "str", "clf": "str", "paradigm": "str", "optim": "str", "lr": "float", "non_iid": "int"}


def line_site(path, line, text):
    return {"path": path, "line_start": line, "line_end": line, "excerpt": text.strip()}


def relative(directory, name):
    path = PurePosixPath(directory, name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("The documented launch has an unsupported working directory or path.")
    return str(path)


def documented_shell(snapshot, stage, architecture):
    target = stage_pattern(stage)
    matches = []
    for path in sorted(snapshot.files):
        if "/" in path or PurePosixPath(path).stem.lower() != "readme":
            continue
        directory, cd, section = ".", None, None
        for number, raw in enumerate(snapshot.read(path).splitlines(), 1):
            line = raw.strip()
            if line.startswith("```"):
                directory, cd, section = ".", None, None
            elif line.startswith("#"):
                section = line_site(path, number, raw)
            elif re.fullmatch(r"cd\s+[\w./-]+", line):
                directory = relative(directory, line.split()[1])
                cd = line_site(path, number, raw)
            else:
                match = re.fullmatch(r"(?:sh|bash)\s+(\S+\.sh)\s+<[\w-]+>\s+<[\w-]+>", line)
                if not match or not section or not describes_stage(section["excerpt"], target):
                    continue
                template = match[1]
                choices = list(CHOICES.finditer(template))
                if len(choices) != 1 or architecture not in choices[0][1].split("|"):
                    continue
                script = relative(directory, CHOICES.sub(architecture, template))
                ref = {**line_site(path, number, raw), "entry_point": script}
                matches.append((ref, directory, [section, *([cd] if cd else [])]))
    return matches


def shell_commands(snapshot, script, directory, architecture, branch, stage):
    if script not in snapshot.files:
        raise ValueError("The selected shell script is absent from the commit.")
    active, functions, top = None, {}, []
    for number, raw in enumerate(snapshot.read(script).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        site = line_site(script, number, raw)
        if active:
            if line == "}":
                active = None
            else:
                functions[active].append(site)
        else:
            definition = FUNCTION.fullmatch(line)
            if definition:
                active = definition[1]
                if active in functions:
                    raise ValueError("The shell script repeats a function definition.")
                functions[active] = [site]
            else:
                top.append(site)
    binding = re.fullmatch(r"([A-Za-z_]\w*)=\$2", top[0]["excerpt"]) if len(top) == 2 else None
    if active or not binding or top[1]["excerpt"] != "$1" or branch not in functions:
        raise ValueError("Could not prove that the shell script dispatches the requested branch and count.")
    commands = []
    for site in functions[branch][1:]:
        raw = site["excerpt"]
        if not re.fullmatch(r"[\w\s./$&+=-]+", raw):
            raise ValueError("The selected shell branch contains an unsupported command.")
        args = shlex.split(raw)
        if args and args[-1] == "&":
            args.pop()
        if len(args) < 4 or not PYTHON.fullmatch(args[0]) or not args[1].endswith(".py"):
            raise ValueError("The selected shell branch contains an unsupported command.")
        options = {}
        current = None
        for arg in args[2:]:
            if arg.startswith("--"):
                if arg in options or not re.fullmatch(r"--[\w-]+", arg):
                    raise ValueError("The selected shell command has ambiguous options.")
                current = arg
                options[current] = []
            elif current is not None and re.fullmatch(r"[\w.+-]+|\$[A-Za-z_]\w*", arg):
                options[current].append(arg)
            else:
                raise ValueError("The selected shell command has unsupported arguments.")
        if not all(options.values()):
            raise ValueError("The selected shell command has an option without a value.")
        regime = options.get("--non-iid")
        if regime != ["$" + binding[1]]:
            raise ValueError("The shell command does not pass the selected non-IID count.")
        paradigm = options.get("--paradigm", [])
        if (options.get("--dataset") != [branch] or options.get("--clf") != [architecture]
                or len(paradigm) != 1 or not stage_pattern(paradigm[0]).search(stage)
                or options.get("--optim") != ["sgd"]):
            raise ValueError("A command in the selected branch cannot be assigned to this SGD experiment.")
        rate = options.get("--lr", [])
        if len(rate) != 1 or not re.fullmatch(NUMBER, rate[0]) or normalized(rate[0]) is None:
            raise ValueError("A selected shell command has no unique literal learning rate.")
        commands.append((relative(directory, args[1]), normalized(rate[0]), site, options.keys()))
    if not commands:
        raise ValueError("The selected shell branch has no documented Python launch.")
    return commands, [functions[branch][0], *top]


def nodes(body):
    """Visit executable statements in a scope without descending into definitions."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        yield from nodes(ast.iter_child_nodes(node))


def assigned_calls(body):
    for statement in body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, ast.Call)):
            yield statement.targets[0].id, statement.value, statement


def find_definition(sources, path, name, kind):
    _, body = sources.read(path)
    matches = [(path, n) for n in body if isinstance(n, kind) and n.name == name]
    for node in body:
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        for item in node.names:
            if (item.asname or item.name) != name:
                continue
            imported = sources.module(node.module)
            if imported:
                matches.extend((imported, n) for n in sources.read(imported)[1]
                               if isinstance(n, kind) and n.name == item.name)
    return matches[0] if len(matches) == 1 else None


def parse_namespace(sources, path, call, flags):
    if not isinstance(call.func, ast.Name) or call.args or call.keywords:
        return None
    parser = find_definition(sources, path, call.func.id, ast.FunctionDef)
    if not parser:
        return None
    file, function = parser
    body = function.body
    constructors = [(name, n) for name, ctor, n in assigned_calls(body)
                    if chain(ctor.func) == ("argparse", "ArgumentParser")]
    if len(constructors) != 1 or not isinstance(body[-1], ast.Return):
        return None
    name, declared = constructors[0]
    namespace = body[-1].value
    parser_trace = []
    returned = chain(namespace)
    if returned:
        steps = [(value, stmt) for target, value, stmt in assigned_calls(body) if (target,) == returned]
        if len(steps) != 2 or name_writes(body, returned[0]) != 2:
            return None
        (conversion, conversion_site), (wrapper, wrapper_site) = steps
        if (chain(conversion.func) != ("vars",) or len(conversion.args) != 1 or conversion.keywords
                or not isinstance(wrapper.func, ast.Name) or wrapper.args or len(wrapper.keywords) != 1
                or wrapper.keywords[0].arg is not None or chain(wrapper.keywords[0].value) != returned):
            return None
        struct = find_definition(sources, file, wrapper.func.id, ast.ClassDef)
        if not struct:
            return None
        struct_file, cls = struct
        inits = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        if len(inits) != 1 or len(inits[0].args.args) != 1 or not inits[0].args.kwarg:
            return None
        init = inits[0]
        statements = init.body
        if len(statements) != 1 or not isinstance(statements[0], ast.Expr) or not isinstance(statements[0].value, ast.Call):
            return None
        update = statements[0].value
        if (chain(update.func) != (init.args.args[0].arg, "__dict__", "update") or len(update.args) != 1
                or update.keywords or chain(update.args[0]) != (init.args.kwarg.arg,)):
            return None
        namespace = conversion.args[0]
        parser_trace = [sources.site(file, conversion_site), sources.site(file, wrapper_site),
                        sources.site(struct_file, statements[0])]
    if (not isinstance(namespace, ast.Call) or chain(namespace.func) != (name, "parse_args")
            or namespace.args or namespace.keywords):
        return None
    items = [n.value for n in body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and chain(n.value.func) == (name, "add_argument")]
    definitions = {}
    for option in flags:
        matches = [n for n in items if len(n.args) == 1 and isinstance(n.args[0], ast.Constant)
                   and n.args[0].value == option]
        if len(matches) != 1:
            return None
        definitions[option] = matches[0]
    for field, kind in FIELDS.items():
        declaration = definitions.get("--" + field.replace("_", "-"))
        if not declaration:
            return None
        kw = {k.arg: k.value for k in declaration.keywords}
        if chain(kw.get("type")) != (kind,) or {"dest", "action", None} & kw.keys():
            return None
        if "nargs" in kw and not (field == "paradigm" and isinstance(kw["nargs"], ast.Constant)
                                 and kw["nargs"].value == "+"):
            return None
    return [sources.site(file, declared), *(sources.site(file, definitions["--"+field.replace("_", "-")])
            for field in FIELDS), sources.site(file, namespace), *parser_trace]


def carried_fields(sources, path, wrapper):
    if not isinstance(wrapper.func, ast.Name):
        return None
    resolved = find_definition(sources, path, wrapper.func.id, ast.ClassDef)
    if not resolved:
        return None
    file, cls = resolved
    inits = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if len(inits) != 1 or len(inits[0].args.args) != 2:
        return None
    init = inits[0]
    target, origin = [a.arg for a in init.args.args]
    trace = []
    for field in FIELDS:
        writes = [n for n in nodes(init.body) if isinstance(n, ast.Attribute)
                  and isinstance(n.ctx, (ast.Store, ast.Del)) and chain(n) == (target, field)]
        matches = [n for n in init.body if isinstance(n, ast.Assign)
                   and len(writes) == 1 and n.targets == writes and chain(n.value) == (origin, field)]
        if len(matches) != 1:
            return None
        trace.append(sources.site(file, matches[0]))
    return trace


def optimizer_sites(sources, path, body, name, trace, active, selected=False):
    identified = []
    for node in nodes(body):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if chain(node) in {(name, field) for field in FIELDS}:
                raise ValueError("A selected command's parsed arguments can change before SGD.")
    for statement in body:
        if isinstance(statement, ast.If):
            condition = statement.test
            guard = (isinstance(condition, ast.Compare) and len(condition.ops) == 1
                     and isinstance(condition.ops[0], ast.Eq) and chain(condition.left) == (name, "optim")
                     and isinstance(condition.comparators[0], ast.Constant))
            if guard:
                same = condition.comparators[0].value == "sgd"
                identified.extend(optimizer_sites(sources, path, statement.body if same else statement.orelse,
                                                  name, [*trace, sources.site(path, condition)], active,
                                                  selected or same))
                continue
        # Scan a control-flow block for call sites carrying the same namespace.
        for call in nodes([statement]):
            if not isinstance(call, ast.Call):
                continue
            target = chain(call.func)
            if selected and target and target[-1] == "SGD":
                imports = [n for n in sources.read(path)[1] if isinstance(n, ast.Import)
                           and any(item.name == "torch.optim"
                                   and tuple((item.asname or item.name).split(".")) == target[:-1]
                                   for item in n.names)]
                rate = [kw.value for kw in call.keywords if kw.arg == "lr"]
                if imports and len(rate) == 1 and chain(rate[0]) == (name, "lr") and all(
                    kw.arg is not None for kw in call.keywords
                ):
                    identified.append((sources.site(path, call), trace))
            if len(target) != 1 or not call.args or chain(call.args[0]) != (name,):
                continue
            resolved = find_definition(sources, path, target[0], ast.FunctionDef)
            if resolved:
                file, function = resolved
                marker = (file, function.lineno)
                if marker not in active and function.args.args and len(active) < 6:
                    param = function.args.args[0].arg
                    if not name_writes(function.body, param):
                        identified.extend(optimizer_sites(sources, file, function.body, param,
                                                          [*trace, sources.site(path, call)], {*active, marker}))
    return identified


def shell_lr(snapshot, stage, recipe):
    architecture, branch, count = recipe
    routes = documented_shell(snapshot, stage, architecture)
    if len(routes) != 1:
        return [], [r[0] for r in routes], "The stage does not identify one documented shell launch."
    ref, directory, readme_trace = routes[0]
    try:
        commands, dispatch = shell_commands(snapshot, ref["entry_point"], directory, architecture, branch, stage)
        signatures = {(path, frozenset(flags)) for path, _, _, flags in commands}
        if len(signatures) != 1:
            raise ValueError("The selected shell commands do not share a single Python entry point and option shape.")
        path, flags = signatures.pop()
        if path not in snapshot.files:
            raise ValueError("The Python entry point selected by the shell branch is absent from the commit.")
        sources = Sources(snapshot)
        _, body = sources.read(path)
        traces = []
        for namespace, call, statement in assigned_calls(body):
            if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Call):
                continue
            parse = parse_namespace(sources, path, call.args[0], flags)
            fields = carried_fields(sources, path, call) if parse else None
            if parse and fields and name_writes(body, namespace) == 1:
                traces.extend(optimizer_sites(sources, path, body, namespace,
                                              [sources.site(path, statement), *parse, *fields], set()))
        if len(traces) != 1:
            raise ValueError("Could not link the parsed shell LR to one local SGD optimizer.")
        use, trail = traces[0]
        return [
            {"value": value, "use": use, "definitions": [site],
             "stage_trace": [*readme_trace, *dispatch, *trail],
             "non_iid_count": int(count), "shell_branch": branch, "architecture": architecture}
            for _, value, site, _ in commands
        ], [ref], None
    except (ValueError, SyntaxError) as error:
        return [], [ref], str(error)
