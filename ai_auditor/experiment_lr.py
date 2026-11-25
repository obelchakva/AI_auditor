"""Explicit experiment context and documented argparse recipes."""

import ast
import re
import shlex
import unicodedata

from .flag_lr import Sources, chain, launch_options
from .initial_lr import (
    NUMBER, OPTIMIZER_LR, PAPER_LR, compact, evidence, literal, lr_expressions,
    main_body, normalized, open_article,
)


INITIALIZE = re.compile(
    rf"\binitiali[sz]e\s+(?:the\s+)?learning rate\s+(?:as|to)\s+({NUMBER})(?![\w+%/]|\.\d)",
    re.I,
)


def clean(text):
    return compact(unicodedata.normalize("NFKC", text)).casefold()


def subject_key(text):
    return re.sub(r"[^\w]", "", clean(text))


def table_pair(page, blocks, context, model_pair, table):
    teacher, student = map(clean, model_pair)
    words = page.get_text("words")
    captions = sorted((b for b in blocks if re.match(r"^Table\s+\d+\s*[:.]", b[4], re.I)),
                      key=lambda b: b[1])
    found = []
    for top in words:
        if clean(top[4]) != teacher:
            continue
        for bottom in words:
            if (clean(bottom[4]) != student or not 0 < bottom[1] - top[1] <= 24
                    or abs((top[0] + top[2]) - (bottom[0] + bottom[2])) > 12):
                continue
            def row_label(label, model):
                return any(clean(w[4]) == label and w[0] < model[0] and abs(w[1] - model[1]) <= 3
                           for w in words)
            caption = next((b for b in captions if b[1] > bottom[3] and b[0] <= bottom[0] <= b[2]), None)
            if (caption and (table is None or re.match(rf"^Table\s+{table}\s*[:.]", caption[4], re.I))
                    and subject_key(context) in subject_key(caption[4])
                    and row_label("teacher", top) and row_label("student", bottom)):
                item = {"page": page.number + 1, "excerpt": f"Teacher {top[4]} / Student {bottom[4]}",
                        "table_caption": compact(caption[4])}
                if item not in found:
                    found.append(item)
    return found


def section_candidates(pdf, section, context, model_pair, table):
    heading = clean(section)
    subject = re.compile(rf"(?<!\w){re.escape(clean(context))}(?!\w)")
    candidates, partial = [], []
    pair = []
    active = False
    with open_article(pdf) as article:
        for page in article:
            blocks = [block for block in page.get_text("blocks") if block[6] == 0]
            if model_pair:
                pair.extend(table_pair(page, blocks, context, model_pair, table))
            for block in blocks:
                lines = block[4].strip().splitlines()
                if not lines:
                    continue
                label = clean(lines[0])
                if re.fullmatch(r"\d+(?:\.\d+)*", label) and len(lines) > 1:
                    label += " " + clean(lines[1])
                if label == heading:
                    active = True
                    partial.append({"page": page.number + 1, "excerpt": compact(block[4])})
                    continue
                if re.match(r"^\d+(?:\.\d+)+\s+\D|^\d+\s+[A-Z]", compact(block[4])):
                    active = False
                if not active or not subject.search(clean(block[4])):
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", compact(block[4])):
                    if not subject.search(clean(sentence)):
                        continue
                    for pattern in (PAPER_LR, OPTIMIZER_LR, INITIALIZE):
                        for match in pattern.finditer(sentence):
                            value = normalized(match[1])
                            if value is not None:
                                item = {"page": page.number + 1, "excerpt": compact(block[4])}
                                partial.append(item)
                                # An explicitly named student with its own LR in this
                                # paragraph needs a separate comparison.
                                if model_pair and any(
                                    clean(model_pair[1]) in clean(other) and "learning rate" in clean(other)
                                    for other in re.split(r"(?<=[.!?])\s+", compact(block[4])) if other != sentence
                                ):
                                    continue
                                candidates.append({**item, "value": value, "section": section})
    if model_pair and not pair:
        return [], partial
    if pair:
        for item in candidates:
            item["model_pair_evidence"] = pair
    return candidates, partial + pair


EXPLICIT_SCRIPT = re.compile(
    r"python(?:3(?:\.\d+)?)?\s+(?:-u\s+)?(?:\./)?(?P<script>[\w./-]+\.py)(?![\w./-])"
)


def dotlist_options(tail):
    """Read scalar and flat list overrides without expanding shell syntax."""
    try:
        tokens = shlex.split(tail, comments=True)
    except ValueError:
        return None
    options = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if (not sep or key in options or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", key)
                or not re.fullmatch(r"[\w./+-]+|\[(?:[\w./+-]+(?:,[\w./+-]+)*)?\]", value)):
            return None
        options[key] = value
    return options or None


def readme_recipes(snapshot, criteria, context):
    recipes = []
    for path in sorted(snapshot.files):
        if "/" in path or not re.fullmatch(r"readme(?:\.[\w-]+)?", path, re.I):
            continue
        prose = ""
        inside_fence = False
        for number, line in enumerate(snapshot.read(path).splitlines(), 1):
            if line.strip().startswith("```"):
                inside_fence = not inside_fence
                continue
            command = EXPLICIT_SCRIPT.match(line.lstrip().lstrip("`"))
            if not command or not command["script"]:
                if not inside_fence and line.strip():
                    prose = line.strip()
                continue
            command_line = line.lstrip().lstrip("`")
            tail = command_line[command.end():]
            options = launch_options(tail, allow_short=True)
            style = "argparse"
            if options is None:
                options = dotlist_options(tail.rstrip().removesuffix("`"))
                style = "hydra"
            if options is None or any(options.get(flag) != value for flag, value in criteria):
                continue
            if context and clean(context) not in clean(prose):
                continue
            recipes.append({"path": path, "line_start": number, "line_end": number,
                            "excerpt": line.strip(), "context": prose,
                            "entry_point": command["script"], "arguments": options, "argument_style": style})
    return recipes


def static_value(node, namespace, values):
    if isinstance(node, ast.Constant):
        return node.value
    ref = chain(node)
    if len(ref) == 2 and ref[0] == namespace:
        return values.get(ref[1])
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(static_value(item, namespace, values) for item in node.elts)
    return None


def condition(node, namespace, values):
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = static_value(node.left, namespace, values)
        right = static_value(node.comparators[0], namespace, values)
        if left is None or right is None:
            return None
        op = node.ops[0]
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, tuple) and None not in right:
            return (left in right) != isinstance(op, ast.NotIn)
        if isinstance(op, (ast.Eq, ast.NotEq)):
            return (left == right) != isinstance(op, ast.NotEq)
    return None


def declared_options(function, options):
    parsers = [node.targets[0].id for node in function.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and isinstance(node.value, ast.Call) and chain(node.value.func) == ("argparse", "ArgumentParser")]
    if len(parsers) != 1:
        raise ValueError("The argument parser is not uniquely resolved.")
    linear_calls = {node.value for node in function.body
                    if isinstance(node, (ast.Expr, ast.Assign)) and isinstance(node.value, ast.Call)}
    parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
    for item in ast.walk(function):
        if not isinstance(item, ast.Name) or item.id != parsers[0]:
            continue
        if isinstance(item.ctx, (ast.Store, ast.Del)):
            if not (isinstance(parents[item], ast.Assign)
                    and isinstance(parents[item].value, ast.Call)
                    and chain(parents[item].value.func) == ("argparse", "ArgumentParser")):
                raise ValueError("The argument parser can be reassigned.")
            continue
        parent = parents[item]
        if not (isinstance(parent, ast.Attribute) and parent.attr in {"add_argument", "parse_args"}
                and parents[parent] in linear_calls and parents[parent].func is parent):
            raise ValueError("Unsupported use of the argument parser could change its defaults.")
    declarations = {}
    values = {}
    remaining = dict(options)
    for node in function.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if chain(call.func) != (parsers[0], "add_argument"):
            continue
        flags = [arg.value for arg in call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        keywords = {kw.arg: kw.value for kw in call.keywords}
        long_flag = next((flag for flag in flags if flag.startswith("--")), "")
        if not long_flag or set(keywords) & {"action", "nargs", None}:
            continue
        dest = static_value(keywords.get("dest"), "", {}) or long_flag[2:].replace("-", "_")
        chosen = [remaining.pop(flag) for flag in flags if flag in remaining]
        if dest in declarations or len(chosen) > 1:
            raise ValueError("An argument declaration or override is ambiguous.")
        converter = chain(keywords.get("type"))
        if converter not in {(), ("str",), ("float",), ("int",)} or ("type" in keywords and not converter):
            if chosen:
                raise ValueError("The selected command uses an unsupported argument converter.")
            continue
        declarations[dest] = (call, chosen[0] if chosen else None, converter)
        if chosen:
            raw = chosen[0]
            values[dest] = float(raw) if converter == ("float",) else int(raw) if converter == ("int",) else raw
        else:
            values[dest] = static_value(keywords.get("default"), "", {})
    if remaining:
        raise ValueError("The selected command contains options without supported declarations.")
    return parsers[0], declarations, values


def calls_with_namespace(node, namespace):
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and any(
            chain(arg) == (namespace,) for arg in [*call.args, *(arg.value for arg in call.keywords)]
        ):
            yield call


def read_only_constructor(sources, body, call, namespace):
    name = chain(call.func)
    if len(name) != 1 or len(call.args) != 1 or chain(call.args[0]) != (namespace,) or call.keywords:
        return None
    for node in body:
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name[0]:
                continue
            module = sources.module(node.module)
            if not module:
                return None
            for cls in sources.read(module)[1]:
                if not isinstance(cls, ast.ClassDef) or cls.name != alias.name:
                    continue
                methods = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"]
                if len(methods) != 1 or len(methods[0].args.args) != 2:
                    return None
                method = methods[0]
                arg = method.args.args[1].arg
                parents = {child: parent for parent in ast.walk(method) for child in ast.iter_child_nodes(parent)}
                for use in ast.walk(method):
                    if isinstance(use, ast.Name) and use.id == arg:
                        parent = parents[use]
                        if (not isinstance(parent, ast.Attribute) or not isinstance(parent.ctx, ast.Load)
                                or parent.attr.startswith("__")
                                or isinstance(parents.get(parent), ast.Call) and parents[parent].func is parent):
                            return None
                return sources.site(module, method)
    return None


def trace_updates(statements, namespace, rate, values, sources, path, module, candidate):
    for statement in statements:
        if isinstance(statement, ast.If):
            branch = condition(statement.test, namespace, values)
            targets = statement.body if branch else statement.orelse
            if branch is not None:
                if any(chain(node) == (namespace, rate) and isinstance(node.ctx, (ast.Store, ast.Del))
                       for node in ast.walk(statement)):
                    candidate["stage_trace"].append(sources.site(path, statement))
                trace_updates(targets, namespace, rate, values, sources, path, module, candidate)
                continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = chain(statement.targets[0])
            if len(target) == 2 and target[0] == namespace:
                values[target[1]] = static_value(statement.value, namespace, values)
                if target[1] == rate:
                    candidate["value"] = literal(sources.read(path)[0], statement.value)
                    candidate["definitions"].append(sources.site(path, statement))
        for node in ast.walk(statement):
            if isinstance(node, (ast.Return, ast.Raise)):
                raise ValueError("The selected option flow can exit before optimizer creation.")
            if isinstance(node, ast.Name) and node.id == namespace:
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    raise ValueError("The option namespace is reassigned before optimizer creation.")
            if isinstance(node, ast.Attribute) and chain(node.value) == (namespace,) and node.attr.startswith("__"):
                raise ValueError("The option namespace is accessed through dynamic attributes.")
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                    and len(chain(node)) == 2 and chain(node)[0] == namespace):
                if not isinstance(statement, ast.Assign) or node not in statement.targets:
                    if node.attr == rate:
                        candidate["stage_trace"].append(sources.site(path, statement))
                        raise ValueError("An unresolved code path can change the LR before optimizer creation.")
                    values.pop(node.attr, None)
        for call in calls_with_namespace(statement, namespace):
            proof = read_only_constructor(sources, module, call, namespace)
            if proof is None:
                raise ValueError("The option namespace reaches a call that could change the LR.")
            candidate["stage_trace"].extend([sources.site(path, call), proof])
        parents = {child: parent for parent in ast.walk(statement) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == namespace:
                if not isinstance(parents.get(node), (ast.Attribute, ast.Call, ast.keyword)):
                    raise ValueError("The option namespace can escape static tracking.")


def argparse_recipe_lr(snapshot, reference):
    path = reference["entry_point"]
    sources = Sources(snapshot)
    _, module = sources.read(path)
    functions = {item.name: item for item in module if isinstance(item, ast.FunctionDef)}
    entries = [item.value.func.id for node in module for item in main_body(node)
               if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
               and isinstance(item.value.func, ast.Name) and not item.value.args and not item.value.keywords]
    if len(entries) != 1 or entries[0] not in functions:
        return [], "The selected script lacks a unique supported main call."
    entry = functions[entries[0]].body
    optimizers = [(idx, node.value) for idx, node in enumerate(entry)
                  if isinstance(node, (ast.Assign, ast.Expr)) and isinstance(node.value, ast.Call)
                  and chain(node.value.func)[-1:] == ("SGD",)]
    if len(optimizers) != 1:
        return [], "The selected script lacks a unique straight-line SGD call."
    index, optimizer = optimizers[0]
    rates = list(lr_expressions(optimizer))
    ref = chain(rates[0]) if len(rates) == 1 else ()
    candidate = {"value": None, "use": sources.site(path, optimizer), "definitions": [], "stage_trace": []}
    try:
        if len(ref) != 2:
            raise ValueError("SGD does not receive one supported option-backed LR.")
        namespace, rate = ref
        [binding] = [node for node in entry[:index] if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and chain(node.targets[0]) == (namespace,) and isinstance(node.value, ast.Call)
                     and len(chain(node.value.func)) == 1 and not node.value.args and not node.value.keywords]
        parse = functions.get(binding.value.func.id)
        if parse is None:
            raise ValueError("The option parser is not defined in the selected script.")
        parser, declarations, values = declared_options(parse, reference["arguments"])
        [parsed] = [node for node in parse.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and chain(node.value.func) == (parser, "parse_args") and not node.value.args
                    and not node.value.keywords and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)]
        result = parsed.targets[0].id
        if not isinstance(parse.body[-1], ast.Return) or chain(parse.body[-1].value) != (result,):
            raise ValueError("The parser does not return its option namespace directly.")
        declaration, override, converter = declarations[rate]
        candidate["definitions"].append(sources.site(path, declaration))
        candidate["stage_trace"].extend(sources.site(path, node) for node in (parsed, binding))
        if declaration.lineno >= parsed.lineno:
            raise ValueError("The LR option is declared after the arguments are parsed.")
        if override is not None and converter not in {("float",), ("int",)}:
            raise ValueError("The selected LR override is not parsed as a number.")
        default = next((kw.value for kw in declaration.keywords if kw.arg == "default"), None)
        candidate["value"] = normalized(override) if override is not None else literal(sources.read(path)[0], default)
        if override is not None:
            candidate["command_override"] = override
        trace_updates(parse.body[parse.body.index(parsed) + 1:-1], result, rate, values, sources, path, module, candidate)
        trace_updates(entry[entry.index(binding) + 1:index], namespace, rate, values, sources, path, module, candidate)
        if candidate["value"] is None:
            raise ValueError("The selected recipe does not resolve to a numeric LR.")
    except (ValueError, KeyError, SyntaxError) as error:
        candidate["value"] = None
        return [candidate], str(error) if isinstance(error, (ValueError, SyntaxError)) else "SGD's LR has no supported argument declaration."
    return [candidate], None
