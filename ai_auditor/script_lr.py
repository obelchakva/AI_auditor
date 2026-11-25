"""Check a selected Python script against the source of a generated input."""

import ast
import re
from pathlib import PurePosixPath

from .experiment_lr import clean, declared_options
from .flag_lr import Sources, chain
from .initial_lr import (
    NUMBER, OPTIMIZERS, code_candidates, compact, is_optimizer, literal, lr_expressions,
    main_body, normalized, open_article, straight_line_calls,
)
from .paper_scope import term


def paper_input_rate(pdf, section, context, identity):
    # A general starting LR and the variant must occur in the same
    # implementation paragraph. A separate data paragraph must name the
    # selected input and describe the same data variation.
    rate = re.compile(
        rf"\b({'|'.join(sorted(OPTIMIZERS))})\s+optimizer\b[^.!?]*?"
        rf"\bstarting with a learning rate of\s+({NUMBER})(?![\w+/%]|\.\d)", re.I
    )
    subject, dataset = term(context), term(identity)
    keywords = variant_words(context)
    headings, candidates, datasets = [], [], []
    with open_article(pdf) as article:
        for page in article:
            blocks = [compact(re.sub(r"(?<=[^\W\d_])-\n(?=[^\W\d_])", "", block[4]))
                      for block in page.get_text("blocks") if block[6] == 0]
            for index, block in enumerate(blocks):
                if clean(block) != clean(section):
                    continue
                headings.append({"page": page.number + 1, "excerpt": block})
                for paragraph in blocks[index + 1:]:
                    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z]\.)\s+[A-Z]", paragraph):
                        break
                    if not subject.search(clean(paragraph)):
                        continue
                    matches = list(rate.finditer(paragraph))
                    for match in matches:
                        value = normalized(match[2])
                        if value is not None:
                            candidates.append({"page": page.number + 1, "excerpt": paragraph,
                                               "value": value, "optimizer": match[1], "section": section})
            for block in blocks:
                if dataset.search(clean(block)) and keywords & variant_words(block):
                    datasets.append({"page": page.number + 1, "excerpt": block})
    partial = headings + candidates + datasets
    datasets = [item for item in datasets if any(item["page"] == match["page"] for match in candidates)]
    if len(headings) != 1 or len(candidates) != 1 or len(datasets) != 1:
        return [], partial
    return [{**candidates[0], "input_evidence": datasets[0], "section_evidence": headings[0],
             "scope_link": "The implementation paragraph states the starting optimizer rate and describes the "
                           "selected variant. The dataset paragraph names the selected input and data variation."}], partial


def variant_words(text):
    return {word.rstrip("s") for word in re.findall(r"[^\W_]+", clean(text)) if len(word) >= 6}


def functions(body):
    result = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            result.setdefault(node.name, []).append(node)
    return {name: matches[0] for name, matches in result.items() if len(matches) == 1}


def entry_block(body):
    blocks = [main_body(node) for node in body if main_body(node)]
    if len(blocks) != 1:
        raise ValueError("The script has no unique Python entry block.")
    return blocks[0]


def bind_input(sources, path, body, options, flag):
    entry = entry_block(body)
    # Reuse the argparse literal/declaration analyzer used for README recipes.
    parser, declarations, _ = declared_options(ast.Module(body=entry, type_ignores=[]), options)
    parsed = [node for node in entry if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
              and chain(node.value.func) == (parser, "parse_args") and not node.value.args and not node.value.keywords]
    if len(parsed) != 1:
        raise ValueError("Cannot uniquely resolve parsing of the selected script arguments.")
    namespace = parsed[0].targets[0].id
    declared = [(name, call) for name, (call, _, converter) in declarations.items()
                if converter in {(), ("str",)} and any(isinstance(arg, ast.Constant) and arg.value == flag for arg in call.args)]
    if len(declared) != 1 or declared[0][1].lineno >= parsed[0].lineno:
        raise ValueError("The selected input flag is not declared as a literal path before parsing.")
    field, declaration = declared[0]
    routes = routes_for(entry[entry.index(parsed[0]) + 1:], functions(body), (namespace, field))
    if len(routes) != 1:
        raise ValueError("Cannot bind the selected input to a unique training entry call.")
    call, function, param = routes[0]
    for stmt in entry[entry.index(parsed[0]) + 1:entry.index(next(
        node for node in entry if isinstance(node, (ast.Assign, ast.Expr)) and node.value is call
    ))]:
        if any(chain(node) in {(namespace,), (namespace, field)} for node in ast.walk(stmt)):
            raise ValueError("The selected input can be changed before the entry call.")
    return function, param, [sources.site(path, node) for node in (declaration, parsed[0], call)]


def routes_for(body, known, ref):
    routes = []
    for call in straight_line_calls(body):
        f = known.get(getattr(call.func, "id", ""))
        if f is None or call.keywords or f.decorator_list or f.args.vararg or f.args.kwarg:
            continue
        args = f.args.posonlyargs + f.args.args
        for arg, parameter in zip(call.args, args):
            if chain(arg) == ref:
                routes.append((call, f, parameter.arg))
    return routes


def open_pickle(statement, mode, path_filter=None):
    if not isinstance(statement, ast.With) or len(statement.items) != 1:
        return None
    entry = statement.items[0]
    call = entry.context_expr
    if (not isinstance(call, ast.Call) or chain(call.func) != ("open",) or len(call.args) != 2 or call.keywords
            or not isinstance(entry.optional_vars, ast.Name) or not isinstance(call.args[1], ast.Constant)
            or call.args[1].value != mode or path_filter and not path_filter(call.args[0])):
        return None
    return call, entry.optional_vars.id


def target_names(target):
    ref = chain(target)
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(target_names(item) for item in target.elts))
    return {ref[0]} if ref else set()


def tainted(expression, tracked):
    return any(isinstance(node, ast.Name) and node.id in tracked for node in ast.walk(expression))


def propagate(statement, tracked):
    if isinstance(statement, ast.Assign) and tainted(statement.value, tracked):
        for target in statement.targets:
            tracked.update(target_names(target))
    elif isinstance(statement, ast.Assign):
        for target in statement.targets:
            if isinstance(target, ast.Name):
                tracked.discard(target.id)


def load_input(sources, path, known, function, parameter, seen=(), optimizer=None):
    if function.name in seen or any(
        chain(node) == (parameter,) and isinstance(node.ctx, (ast.Store, ast.Del))
        for statement in function.body for node in ast.walk(statement)
    ):
        return []
    origin, tracked = [], set()
    routes = routes_for(function.body, known, (parameter,))
    for statement in function.body:
        opened = open_pickle(statement, "rb", lambda node: chain(node) == (parameter,))
        if opened:
            _, handle = opened
            for item in statement.body:
                if (isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                        and chain(item.value.func) == ("pickle", "load") and len(item.value.args) == 1
                        and chain(item.value.args[0]) == (handle,) and not item.value.keywords):
                    for target in item.targets:
                        tracked.update(target_names(target))
                    origin.append([sources.site(path, statement), sources.site(path, item)])
        elif isinstance(statement, ast.Assign):
            replies = [result for call, f, param in routes if statement.value is call
                       for result in load_input(sources, path, known, f, param, (*seen, function.name))]
            if replies:
                for position, trace in replies:
                    for target in statement.targets:
                        if isinstance(target, (ast.Tuple, ast.List)) and position is not None and position < len(target.elts):
                            tracked.update(target_names(target.elts[position]))
                        elif position is None:
                            tracked.update(target_names(target))
                    origin.append([sources.site(path, statement), *trace])
            else:
                propagate(statement, tracked)
        if isinstance(statement, ast.Return) and len(origin) == 1 and statement.value is not None:
            trace = [*origin[0], sources.site(path, statement)]
            if isinstance(statement.value, ast.Tuple):
                return [(i, trace) for i, item in enumerate(statement.value.elts) if tainted(item, tracked)]
            return [(None, trace)] if tainted(statement.value, tracked) else []
        if optimizer and isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(statement.value, ast.Call):
            call = statement.value
            if any(chain(arg) == (optimizer,) for arg in call.args) and any(tainted(arg, tracked) for arg in call.args):
                if len(origin) == 1:
                    return [(None, [*origin[0], sources.site(path, statement)])]
    return []


def plain_import(body, name):
    return any(isinstance(node, ast.Import) and any(alias.name == name and alias.asname is None for alias in node.names)
               for node in body)


def earlier_definitions(body, names, until):
    """Follow local input assignments back from the preparation call."""
    result = []
    for node in reversed(body):
        if node.lineno >= until or not isinstance(node, ast.Assign):
            continue
        assigned = set().union(*(target_names(target) for target in node.targets))
        if not assigned & names:
            continue
        result.append(node)
        names = (names - assigned) | {item.id for item in ast.walk(node.value) if isinstance(item, ast.Name)}
    return list(reversed(result))


def producer_evidence(sources, path, selected_path, transform, context):
    _, body = sources.read(path)
    known = functions(body)
    chosen = known.get(transform)
    if (not chosen or chosen.decorator_list or not plain_import(body, "pickle")
            or not variant_words(transform) & variant_words(context)):
        raise ValueError("The selected preparation function cannot be tied to the paper's data variation.")
    entry = entry_block(body)
    roots = [(call, known[call.func.id]) for call in straight_line_calls(entry)
             if isinstance(call.func, ast.Name) and call.func.id in known]
    found = []
    for entry_call, root in roots:
        for statement in root.body:
            opened = open_pickle(statement, "wb", lambda node: isinstance(node, ast.Constant) and isinstance(node.value, str)
                                 and PurePosixPath(node.value) == PurePosixPath(selected_path))
            if not opened:
                continue
            _, handle = opened
            for dump in straight_line_calls(statement.body):
                if (chain(dump.func) != ("pickle", "dump") or len(dump.args) != 2 or dump.keywords
                        or chain(dump.args[1]) != (handle,) or not isinstance(dump.args[0], ast.Name)):
                    continue
                variable = dump.args[0].id
                previous = [node for node in root.body if node.lineno < statement.lineno
                            and isinstance(node, ast.Assign) and len(node.targets) == 1
                            and chain(node.targets[0]) == (variable,)]
                if not previous:
                    continue
                assignment = previous[-1]
                expr = assignment.value
                if (not isinstance(expr, ast.Call) or chain(expr.func) != (transform,) or len(expr.args) != 1
                        or expr.keywords or chain(expr.args[0]) != (variable,)):
                    continue
                if any(chain(node) == (variable,) and isinstance(node.ctx, (ast.Store, ast.Del))
                       for stmt in root.body if assignment.lineno < stmt.lineno < statement.lineno
                       for node in ast.walk(stmt)):
                    continue
                upstream = earlier_definitions(root.body, {variable}, assignment.lineno)
                found.append([sources.site(path, entry_call),
                              *(sources.site(path, node) for node in [*upstream, assignment]),
                              sources.site(path, chosen), sources.site(path, statement), sources.site(path, dump)])
    if len(found) != 1:
        raise ValueError("Cannot trace a unique preparation and write of the selected input path on this commit.")
    return found[0]


def training_schedule(sources, path, body, function, optimizer):
    """Recognize a passed optimizer and an explicit per-epoch multiplication."""
    bindings = [node for node in function.body if isinstance(node, ast.Assign) and node.value is optimizer
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)]
    if len(bindings) != 1:
        return []
    variable = bindings[0].targets[0].id
    classes = {node.name: node for node in body if isinstance(node, ast.ClassDef)}
    for node in function.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)):
            continue
        cls = classes.get(getattr(node.value.func, "id", ""))
        if not cls or node.value.keywords:
            continue
        ctor = functions(cls.body).get("__init__")
        if not ctor:
            continue
        incoming = [(parameter.arg, arg) for arg, parameter in zip(node.value.args, ctor.args.args[1:])
                    if chain(arg) == (variable,)]
        if len(incoming) != 1:
            continue
        opt_binding = [item for item in ctor.body if isinstance(item, ast.Assign) and len(item.targets) == 1
                       and isinstance(item.targets[0], ast.Attribute) and chain(item.value) == (incoming[0][0],)]
        if len(opt_binding) != 1:
            continue
        for call in straight_line_calls(function.body):
            method_name = chain(call.func)
            if len(method_name) != 2 or method_name[0] != node.targets[0].id or call.keywords:
                continue
            method = functions(cls.body).get(method_name[1])
            if not method:
                continue
            for arg, formal in zip(call.args, method.args.args[1:]):
                if literal(sources.read(path)[0], arg) is None:
                    continue
                for loop in method.body:
                    if not isinstance(loop, ast.For) or not isinstance(loop.iter, ast.Call) or chain(loop.iter.func) != ("range",):
                        continue
                    for change in loop.body:
                        if not (isinstance(change, ast.Assign) and len(change.targets) == 1
                                and isinstance(change.value, ast.BinOp) and isinstance(change.value.op, ast.Mult)
                                and chain(change.value.right) == (formal.arg,)):
                            continue
                        target = change.targets[0]
                        if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                                and target.slice.value == "lr" and isinstance(target.value, ast.Subscript)
                                and chain(target.value.value) == (method.args.args[0].arg,
                                                                  opt_binding[0].targets[0].attr, "param_groups")
                                and ast.unparse(target) == ast.unparse(change.value.left)):
                            return [sources.site(path, item) for item in (node, opt_binding[0], call, change)]
    return []


def script_recipe_lr(snapshot, reference, identity, context):
    path = reference["entry_point"]
    sources = Sources(snapshot)
    source, body = sources.read(path)
    options, flag = reference["arguments"], reference["input_flag"]
    selected_path = options[flag]
    trace = []
    candidates = []
    partial = {"value": None, "use": None, "definitions": [], "stage_trace": trace}
    try:
        function, parameter, binding = bind_input(sources, path, body, options, flag)
        trace.extend(binding)
        candidates = code_candidates(snapshot, path)
        for candidate in candidates:
            candidate["stage_trace"] = trace
        bindings = [(node, node.value) for node in function.body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call) and is_optimizer(node.value)]
        if len(candidates) != 1 or len(bindings) != 1 or candidates[0]["value"] is None:
            raise ValueError("The selected script has multiple or unresolved sources for the initial LR.")
        binding, call = bindings[0]
        readings = load_input(sources, path, functions(body), function, parameter,
                              optimizer=binding.targets[0].id)
        if not plain_import(body, "pickle") or len(readings) != 1:
            raise ValueError("The selected input is not uniquely traced from a pickle read into the training setup.")
        trace.extend(readings[0][1])
        artifact = PurePosixPath(selected_path)
        if not term(identity).search(clean(re.sub(r"[_-]", " ", artifact.stem))):
            raise ValueError("The selected input path does not identify the dataset named in the PDF.")
        trace.extend(producer_evidence(sources, reference["producer"], selected_path, reference["transform"], context))
        if chain(call.func)[:-1] != ("torch", "optim") or not plain_import(body, "torch"):
            raise ValueError("Cannot verify the constructor as a supported torch optimizer.")
        candidates[0]["optimizer"] = chain(call.func)[-1]
        candidates[0]["use"] = sources.site(path, call)
        if not candidates[0]["definitions"]:
            candidates[0]["definitions"] = [sources.site(path, next(lr_expressions(call)))]
        schedule = training_schedule(sources, path, body, function, call)
        if schedule:
            candidates[0]["subsequent_lr_change"] = schedule
        candidates[0]["input_path"] = selected_path
    except ValueError as error:
        for candidate in candidates:
            if candidate["value"] is not None:
                candidate["unconfirmed_value"] = candidate["value"]
                candidate["value"] = None
        return candidates or [partial], str(error)
    return candidates, None
