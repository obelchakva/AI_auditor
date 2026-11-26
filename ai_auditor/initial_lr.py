"""Compare an explicitly staged paper claim with a documented Git entry point."""

import ast
import re
import subprocess
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

import pymupdf
import yaml

STAGE = re.compile(r"\b(?:step|stage|phase)\s+\d+\b", re.I)
LAUNCH = re.compile(
    r"\bpython(?:3(?:\.\d+)?)?\s+(?:-m\s+(?P<module>\w+(?:\.\w+)*)"
    r"|(?:\./)?(?P<script>[\w./-]+\.py))(?![\w./-])"
)
NUMBER = r"\+?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
PAPER_LR = re.compile(
    r"\b(?:initial\s+(?:learning rate|lr)\s*(?:of|is|was|=|:)\s*"
    r"|(?:learning rate|lr)\s+(?:(?:is|was)\s+)?initially\s+(?:set to\s+)?"
    r"|(?:learning rate|lr)\s+starts?\s+at\s+)"
    rf"({NUMBER})(?![\w+%/]|\.\d)",
    re.I,
)
OPTIMIZER_LR = re.compile(
    rf"\bhyperparameters\s*:\s*(?:learning rate|lr)\s*=\s*({NUMBER})(?![\w+%/]|\.\d)", re.I
)
EPOCHS = re.compile(r"\b(?:for|of)\s+(\d+)\s+epochs\b", re.I)
OPTIMIZERS = {"SGD", "Adam", "AdamW", "Adagrad", "RMSprop"}


def compact(text):
    return " ".join(text.split())


def normalized(raw):
    try:
        number = Decimal(raw.replace("_", ""))
    except InvalidOperation:
        return None
    # Bound formatting for unusual input such as 1e-100000000.
    if not number.is_finite() or not -100 <= number.adjusted() <= 100:
        return None
    expanded = format(number, "f")
    return expanded.rstrip("0").rstrip(".") if "." in expanded else expanded


class Snapshot:
    def __init__(self, repo, commit):
        if not re.fullmatch(r"[a-fA-F0-9]{7,64}", commit):
            raise ValueError("commit must be a hexadecimal Git object ID")
        self.repo = repo
        self.commit = self.git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
        if not self.commit.lower().startswith(commit.lower()):
            raise ValueError("object ID must identify the commit itself")
        self.files = set()
        for item in self.git("ls-tree", "-rz", self.commit).split(b"\0"):
            if not item:
                continue
            metadata, path = item.split(b"\t", 1)
            if metadata.split()[0] in {b"100644", b"100755"}:
                self.files.add(path.decode("utf-8", errors="surrogateescape"))

    def git(self, *args):
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(self.repo), *args],
            capture_output=True, check=False,
        )
        if result.returncode:
            raise ValueError(result.stderr.decode(errors="replace").strip())
        return result.stdout

    def read(self, path):
        return self.git("cat-file", "blob", f"{self.commit}:{path}").decode("utf-8-sig", errors="replace")


def stage_pattern(stage):
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, stage.split())) + r"(?!\w)", re.I)


def describes_stage(text, target):
    return bool(target.search(text)) and all(target.fullmatch(label.group()) for label in STAGE.finditer(text))


@contextmanager
def open_article(pdf):
    with pymupdf.open(pdf) as document:
        if not document.is_pdf or document.needs_pass:
            raise ValueError("article must be a readable, unencrypted PDF")
        yield document


def paper_candidates(pdf, target):
    candidates = []
    with open_article(pdf) as document:
        for page in document:
            # Preserve text blocks so that nearby columns do not interleave.
            for block in page.get_text("blocks"):
                if block[6] != 0 or not block[4].strip():
                    continue
                first_line = block[4].splitlines()[0].strip()
                # A run-in heading on its own PDF line scopes the current
                # text block. An inline stage mention scopes one neighbor.
                heading = first_line if target.fullmatch(first_line.rstrip(".:")) else ""
                previous = ""
                for sentence in re.split(r"(?<=[.!?])\s+", compact(block[4])):
                    labels = list(STAGE.finditer(sentence))
                    parts = [sentence]
                    if len(labels) > 1:
                        starts = [label.start() for label in labels] + [len(sentence)]
                        parts = [sentence[a:b] for a, b in zip(starts, starts[1:])]
                    if labels and not describes_stage(sentence, target):
                        heading = ""
                    for part in parts:
                        direct = describes_stage(part, target)
                        # A short standalone label terminates inherited context.
                        if not direct and re.fullmatch(r"[\w -]{2,60}[.:]", part.strip()):
                            if len(part.split()) <= 5:
                                heading = previous = ""
                        adjacent = previous if re.match(
                            r"^(?:(?:the|its)\s+)?(?:initial\s+)?(?:learning rate|lr)\b", part, re.I
                        ) else ""
                        anchor = part if direct else (heading or adjacent) if not labels else ""
                        if not anchor:
                            continue
                        excerpt = part if direct else f"{anchor} … {part}"
                        for pattern in (PAPER_LR, OPTIMIZER_LR):
                            for match in pattern.finditer(part):
                                value = normalized(match[1])
                                if value is not None:
                                    candidates.append({"value": value, "page": page.number + 1, "excerpt": excerpt})
                    previous = sentence if describes_stage(sentence, target) else ""
    return candidates


def module_paths(snapshot, name, *, entry=True):
    relative = name.replace(".", "/")
    package_file = "__main__.py" if entry else "__init__.py"
    suffixes = (f"{relative}.py", f"{relative}/{package_file}")
    return sorted(path for path in snapshot.files if any(
        path == suffix or path.endswith(f"/{suffix}") for suffix in suffixes
    ))


def entry_points(snapshot, target, stage):
    from .flag_lr import launch_options, selector_matches

    references = []
    for path in sorted(snapshot.files):
        if "/" in path or PurePosixPath(path).stem.lower() != "readme":
            continue
        source = snapshot.read(path).splitlines()
        heading = ""
        section_start = 0
        for index, line in enumerate(source):
            if re.match(r"^\s*#{1,6}\s", line):
                heading, section_start = line, index
            context = line if (target.search(line) or STAGE.search(line)) else heading
            for launch in LAUNCH.finditer(line):
                name = launch["module"]
                related_module = (
                    name and stage_pattern(name.split(".")[-1]).search(heading)
                    and stage_pattern(name.split(".")[-1]).search(stage)
                )
                options = launch_options(line[launch.end():]) if launch["script"] else None
                experiment_options = selector_matches(stage, options)
                if not describes_stage(context, target) and not related_module and not experiment_options:
                    continue
                reference = {"path": path, "line_start": index + 1, "line_end": index + 1,
                             "excerpt": line.strip()}
                if name:
                    paths = module_paths(snapshot, name)
                    reference.update(module=name, module_candidates=paths,
                                     entry_point=paths[0] if len(paths) == 1 else None)
                    tail = line[launch.end():]
                    selector = re.match(r"\s+--(\w[\w-]*)\s+(\w+)\s*(?:[`*]|$)", tail)
                    if selector:
                        flag, placeholder = selector.groups()
                        options = []
                        for row_number, row in enumerate(source[index + 1:], index + 2):
                            if re.match(r"^\s*#", row):
                                break
                            option = re.match(r"^\s*\|\s*([\w-]+)\s*\|", row)
                            if option and option[1] != placeholder and set(option[1]) != {"-"}:
                                options.append({"value": option[1], "line": row_number, "excerpt": row.strip()})
                        if options:
                            reference["selector"] = {"flag": flag, "options": options}
                    if re.match(r"\s+-", tail) and "selector" not in reference:
                        reference["unresolved_arguments"] = True
                    if related_module and not describes_stage(context, target) and "selector" not in reference:
                        continue
                    if related_module:
                        reference["heading"] = {"line": section_start + 1, "excerpt": heading}
                else:
                    reference["entry_point"] = launch["script"]
                    if experiment_options:
                        reference.update(arguments=options, experiment_options=experiment_options)
                references.append(reference)
    return references


def evidence(path, source, node):
    return {"path": path, "line_start": node.lineno, "line_end": node.end_lineno,
            "excerpt": ast.get_source_segment(source, node)}


def assignments(body, name):
    return [statement for statement in body if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name]


def name_writes(body, name):
    return sum(isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name
               for statement in body for node in ast.walk(statement))


def literal(source, node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return normalized(ast.get_source_segment(source, node))
    return None


def imported_constant(snapshot, path, module, expression):
    if not isinstance(expression, ast.Attribute) or not isinstance(expression.value, ast.Name):
        return None
    alias = expression.value.id
    imports = [(statement.module, item.name) for statement in module.body
               if isinstance(statement, ast.ImportFrom) and not statement.level and statement.module
               for item in statement.names if (item.asname or item.name) == alias]
    if len(imports) != 1 or name_writes(module.body, alias):
        return None
    module_name, object_name = imports[0]
    if "." in module_name:
        return None
    config_path = str(PurePosixPath(path).parent / f"{module_name}.py")
    if config_path not in snapshot.files:
        return None
    source = snapshot.read(config_path)
    body = ast.parse(source).body
    objects = assignments(body, object_name)
    if len(objects) != 1 or name_writes(body, object_name) != 1:
        return None
    initializer = objects[0].value
    if (not isinstance(initializer, ast.Call) or not isinstance(initializer.func, ast.Name)
            or initializer.args or initializer.keywords):
        return None
    classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == initializer.func.id]
    if len(classes) != 1 or classes[0].bases or classes[0].keywords:
        return None
    class_body = classes[0].body
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in class_body):
        return None
    constants = assignments(class_body, expression.attr)
    if len(constants) != 1 or name_writes(class_body, expression.attr) != 1:
        return None
    node = constants[0]
    return literal(source, node.value), [evidence(config_path, source, node)]


def main_body(node):
    if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"):
        return node.body
    return []


def straight_line_calls(body):
    for statement in body:
        if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(statement.value, ast.Call):
            yield statement.value


def is_optimizer(call):
    name = call.func
    return (
        isinstance(name, ast.Attribute) and name.attr in OPTIMIZERS
        or isinstance(name, ast.Name) and name.id in OPTIMIZERS
    )


def lr_expressions(call):
    # Handle explicit optimizer-wide lr and literal per-group overrides.
    fallback = next((keyword.value for keyword in call.keywords if keyword.arg == "lr"), None)
    groups = call.args[0] if call.args else next(
        (keyword.value for keyword in call.keywords if keyword.arg == "params"), None
    )
    if any(keyword.arg is None for keyword in call.keywords):
        yield None
    if not isinstance(groups, (ast.List, ast.Tuple)) or not any(
        isinstance(group, ast.Dict) for group in groups.elts
    ):
        yield fallback
        return
    for group in groups.elts:
        if not isinstance(group, ast.Dict) or any(key is None for key in group.keys):
            yield None
            continue
        overrides = [value for key, value in zip(group.keys, group.values)
                     if isinstance(key, ast.Constant) and key.value == "lr"]
        yield overrides[0] if len(overrides) == 1 else fallback if not overrides else None


def code_candidates(snapshot, path):
    source = snapshot.read(path)
    module = ast.parse(source)
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    entry = []
    for node in module.body:
        entry.extend(main_body(node) or [node])
    scopes = [(entry, set())]
    for call in straight_line_calls(entry):
        if isinstance(call.func, ast.Name) and call.func.id in functions:
            function = functions[call.func.id]
            args = function.args
            parameters = {arg.arg for arg in ast.walk(args) if isinstance(arg, ast.arg)}
            scopes.append((function.body, parameters))

    candidates = []
    for body, parameters in scopes:
        direct = [call for call in straight_line_calls(body) if is_optimizer(call)]
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in ast.walk(statement):
                if isinstance(call, ast.Call) and is_optimizer(call) and call not in direct:
                    candidates.append({"value": None, "use": evidence(path, source, call), "definitions": []})
        for call in direct:
            for expression in lr_expressions(call):
                location = evidence(path, source, expression if expression is not None else call)
                value = literal(source, expression)
                definition = []
                if isinstance(expression, ast.Name) and expression.id not in parameters:
                    local = assignments(body, expression.id)
                    scope = body if local or name_writes(body, expression.id) else module.body
                    constants = assignments(scope, expression.id)
                    if len(constants) == 1 and name_writes(scope, expression.id) == 1:
                        constant = constants[0]
                        if constant.lineno < call.lineno or scope is module.body and body is not entry:
                            value = literal(source, constant.value)
                            definition = [evidence(path, source, constant)]
                if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
                    alias = expression.value.id
                    changed = any(
                        isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))
                        and isinstance(n.value, ast.Name) and n.value.id == alias
                        for stmt in (*module.body, *body) for n in ast.walk(stmt)
                    )
                    if alias not in parameters and not changed and not name_writes(body, alias):
                        resolved = imported_constant(snapshot, path, module, expression)
                        if resolved:
                            value, definition = resolved
                candidates.append({"value": value, "use": location, "definitions": definition})
    return candidates


def documented_training(snapshot, reference, stage, article):
    """Trace a documented selector to a matching model's straight-line fit method."""
    path = reference["entry_point"]
    source = snapshot.read(path)
    body = ast.parse(source).body
    selector = reference["selector"]
    options = {option["value"]: option for option in selector["options"]}
    flag = selector["flag"].replace("-", "_")
    component = reference["module"].split(".")[-1].lower()
    words = re.findall(r"\w+", stage.lower())
    model = "".join(word for word in words if word != component)
    names = {component + model, model + component}
    epochs = {int(value) for item in article for value in EPOCHS.findall(item["excerpt"])}
    functions = {node.name: node for node in body if isinstance(node, ast.FunctionDef)}
    imports = {item.asname or item.name: (node.module, item.name)
               for node in body if isinstance(node, ast.ImportFrom) and not node.level and node.module
               for item in node.names if item.name.lower() in names}

    candidates = []
    for guard in body:
        guarded = main_body(guard)
        calls = list(straight_line_calls(guarded))
        declared = any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            and any(isinstance(arg, ast.Constant) and arg.value == f"--{selector['flag']}" for arg in call.args)
            and not any(keyword.arg in {"dest", None} for keyword in call.keywords)
            for call in calls
        )
        namespaces = {
            stmt.targets[0].id for stmt in guarded
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "parse_args" and not stmt.value.args and not stmt.value.keywords
        }
        if not declared or not namespaces:
            continue
        for statement in guarded:
            branch = statement
            while isinstance(branch, ast.If):
                test = branch.test
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Attribute)
                        and isinstance(test.left.value, ast.Name) and test.left.value.id in namespaces
                        and test.left.attr == flag and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq)
                        and isinstance(test.comparators[0], ast.Constant)
                        and isinstance(test.comparators[0].value, str)):
                    option = options.get(test.comparators[0].value)
                    calls = list(straight_line_calls(branch.body))
                    if option and len(calls) == 1 and isinstance(calls[0].func, ast.Name):
                        function = functions.get(calls[0].func.id)
                        if function:
                            candidates.extend(model_training(
                                snapshot, path, source, function, calls[0], test, option, reference, imports, epochs
                            ))
                branch = branch.orelse[0] if len(branch.orelse) == 1 else None
    return candidates


def model_training(snapshot, path, source, function, dispatch, condition, option, reference, imports, epochs):
    if epochs:
        defaults = dict(zip(
            [arg.arg for arg in function.args.args][-len(function.args.defaults):], function.args.defaults
        ))
        count = next((keyword.value for keyword in dispatch.keywords if keyword.arg == "n_epochs"),
                     defaults.get("n_epochs"))
        if len(epochs) != 1 or dispatch.args or not isinstance(count, ast.Constant) or count.value not in epochs:
            return []
    result = []
    for call in straight_line_calls(function.body):
        method = call.func
        if not (isinstance(method, ast.Attribute) and method.attr == "fit" and isinstance(method.value, ast.Name)):
            continue
        variable = method.value.id
        definitions = assignments(function.body, variable)
        if len(definitions) != 1 or name_writes(function.body, variable) != 1:
            continue
        definition = definitions[0]
        constructor = definition.value
        if (definition.lineno >= call.lineno or not isinstance(constructor, ast.Call)
                or not isinstance(constructor.func, ast.Name) or constructor.func.id not in imports):
            continue
        module_name, class_name = imports[constructor.func.id]
        locations = module_paths(snapshot, module_name, entry=False)
        if len(locations) != 1:
            result.append({"value": None, "use": evidence(path, source, definition), "definitions": []})
            continue
        model_path = locations[0]
        model_source = snapshot.read(model_path)
        classes = [cls for cls in ast.parse(model_source).body
                   if isinstance(cls, ast.ClassDef) and cls.name == class_name]
        methods = [member for cls in classes for member in cls.body
                   if isinstance(member, ast.FunctionDef) and member.name == method.attr]
        trace = [
            {"path": reference["path"], "line_start": option["line"], "line_end": option["line"],
             "excerpt": option["excerpt"]},
            evidence(path, source, condition), evidence(path, source, dispatch),
            evidence(path, source, definition), evidence(path, source, call),
        ]
        if len(classes) == 1 and len(methods) == 1:
            result.extend(method_rates(model_path, model_source, methods[0], trace))
    return result


def method_rates(path, source, method, trace):
    direct = list(straight_line_calls(method.body))
    parameters = method.args.posonlyargs + method.args.args
    receiver = parameters[0].arg if parameters else None
    results = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not is_optimizer(node):
            continue
        for expression in lr_expressions(node) if node in direct else [None]:
            value = literal(source, expression)
            definitions = []
            if (isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name)
                    and expression.value.id == receiver):
                writes = [n for n in ast.walk(method)
                          if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))
                          and isinstance(n.value, ast.Name) and n.value.id == receiver and n.attr == expression.attr]
                for statement in method.body:
                    if (len(writes) == 1 and isinstance(statement, ast.Assign)
                            and statement.targets == writes and statement.lineno < node.lineno):
                        value = literal(source, statement.value)
                        definitions = [evidence(path, source, statement)]
            results.append({"value": value, "use": evidence(path, source, node),
                            "definitions": definitions, "stage_trace": trace})
    return results


def audit(pdf, repo, commit, stage, group=None, shell_recipe=None, *, paper_selector=None, recipe_selector=None,
          selected_script=None, table_lr=None, table_shell_recipe=None):
    snapshot = Snapshot(repo, commit)
    target = stage_pattern(stage)
    from .grouped_lr import GROUPS, grouped_code, grouped_paper, words, yaml_launch
    from .flag_lr import flag_lr
    from .experiment_lr import argparse_recipe_lr, readme_recipes, section_candidates

    group = words(group) if group else None
    partial_paper = []
    cross_page = paper_selector and len(paper_selector) == 6 and paper_selector[5]
    table_issue = None
    if table_lr:
        from .paper_lr_table import table_rate

        article, partial_paper, table_issue = table_rate(pdf, *table_lr, include_settings=bool(table_shell_recipe))
    elif selected_script:
        from .script_lr import paper_input_rate

        article, partial_paper = paper_input_rate(pdf, *paper_selector[:2], selected_script[4])
    elif cross_page:
        from .paper_scope import scoped_default

        section, optimizer, _, _, experiment, scope = paper_selector
        article, partial_paper = scoped_default(pdf, section, optimizer, experiment, scope)
    elif paper_selector:
        article, partial_paper = section_candidates(pdf, *paper_selector[:4])
    else:
        article = grouped_paper(pdf, stage, group) if group else paper_candidates(pdf, target)
    if not (paper_selector or group or article or table_lr):
        from .paper_scope import transferred_training_rate

        article, partial_paper = transferred_training_rate(pdf, stage)
    if selected_script:
        script, input_flag, producer, transform, identity, options = selected_script
        references = [{"source": "user_selected_script", "entry_point": script, "arguments": dict(options),
                       "input_flag": input_flag, "producer": producer, "transform": transform}]
    elif table_lr:
        references = []
    elif recipe_selector:
        references = readme_recipes(snapshot, *recipe_selector)
    else:
        references = [] if shell_recipe else entry_points(snapshot, target, stage)
    paths = sorted({ref["entry_point"] for ref in references if ref["entry_point"]})
    code = []
    code_issue = None
    ambiguous_modules = [ref["module"] for ref in references if len(ref.get("module_candidates", [])) > 1]
    missing_modules = [ref["module"] for ref in references if ref.get("module_candidates") == []]
    grouped_launches = [] if selected_script else [ref for ref in references if yaml_launch(ref)]
    code_issue_kind = "code_lr_unresolved"
    if table_shell_recipe:
        from .distributed_lr import table_shell_audit

        code, references, code_issue = table_shell_audit(snapshot, table_shell_recipe, article)
        code_issue_kind = "recipe_scope_unconfirmed" if code else "code_lr_unresolved"
    elif table_lr:
        code_issue_kind = "recipe_not_found"
        code_issue = "No code recipe is verified for the selected table's model/method and dataset."
    elif selected_script:
        from .script_lr import script_recipe_lr

        if script not in snapshot.files or producer not in snapshot.files:
            code_issue_kind = "recipe_not_found"
            code_issue = "The selected script or input producer is absent from the pinned commit."
        else:
            try:
                code, code_issue = script_recipe_lr(snapshot, references[0], identity, paper_selector[1])
            except SyntaxError:
                code_issue = "The selected script or producer is not valid Python."
        if article and code and code[0].get("optimizer") and code[0]["optimizer"].casefold() != article[0]["optimizer"].casefold():
            partial_paper = article + partial_paper
            article = []
    elif recipe_selector:
        if not references:
            code_issue_kind = "recipe_not_found"
            code_issue = "No README command matches the explicit recipe selectors."
        elif len(references) != 1:
            code_issue_kind = "recipe_ambiguous"
            code_issue = "Multiple README commands match the explicit recipe selectors."
        elif paths[0] not in snapshot.files:
            code_issue = "The selected README command points to a script absent from the commit."
        else:
            try:
                reference = references[0]
                if reference["argument_style"] == "hydra":
                    from .hydra_lr import hydra_recipe_lr

                    code, code_issue = hydra_recipe_lr(snapshot, reference)
                else:
                    code, code_issue = argparse_recipe_lr(snapshot, reference)
            except SyntaxError:
                code_issue = "The selected Python recipe could not be parsed."
    elif shell_recipe:
        from .shell_lr import shell_lr

        code, references, code_issue = shell_lr(snapshot, stage, shell_recipe)
    elif group and group not in GROUPS:
        code_issue = "Unknown optimizer group. Supported selectors: " + ", ".join(GROUPS) + "."
    elif group:
        if len(references) != 1 or len(grouped_launches) != 1 or len(paths) != 1 or paths[0] not in snapshot.files:
            code_issue = "The stage does not identify a unique YAML-backed entry point for the selected group."
        else:
            try:
                code = grouped_code(snapshot, grouped_launches[0], group)
            except (ValueError, SyntaxError) as error:
                code_issue = str(error)
            except yaml.YAMLError:
                code_issue = "The selected YAML cannot be parsed safely."
    elif grouped_launches:
        code_issue = "The YAML-backed stage requires an explicit optimizer group (--group)."
    elif ambiguous_modules:
        code_issue_kind = "recipe_ambiguous"
        code_issue = "The selected commit has multiple entry points for module(s): " + ", ".join(ambiguous_modules) + "."
    elif missing_modules:
        code_issue_kind = "recipe_not_found"
        code_issue = "The selected commit has no entry point for module(s): " + ", ".join(missing_modules) + "."
    elif any("experiment_options" in ref for ref in references):
        if len(references) != 1 or len(paths) != 1 or paths[0] not in snapshot.files:
            code_issue = "The requested experiment does not identify a unique documented flag-backed launch."
        else:
            try:
                code = flag_lr(snapshot, references[0])
            except SyntaxError:
                code_issue = "The documented entry point or its model could not be parsed as Python."
            except ValueError as error:
                code_issue = str(error)
    elif any(ref.get("unresolved_arguments") or re.search(r"\.py\s+-{1,2}[\w-]", ref["excerpt"])
             for ref in references):
        code_issue = "The documented launch uses command-line overrides that this check cannot resolve."
    elif len(paths) == 1 and paths[0] in snapshot.files:
        try:
            selectors = [ref for ref in references if "selector" in ref]
            if selectors:
                for reference in selectors:
                    code.extend(documented_training(snapshot, reference, stage, article))
                if not code:
                    code_issue = "No documented module option could be linked to the requested model training."
                elif len({(item["use"]["path"], item["use"]["line_start"]) for item in code}) > 1:
                    code_issue = "Multiple distinct optimizer sites remain for the documented module options."
            else:
                code = code_candidates(snapshot, paths[0])
        except SyntaxError:
            code_issue = "The documented entry point or its configuration could not be parsed as Python."
    elif len(paths) > 1:
        code_issue_kind = "recipe_ambiguous"
        code_issue = "The requested stage has multiple documented Python entry points."
    elif paths:
        code_issue = "The documented Python entry point is absent from the selected commit."
    else:
        code_issue_kind = "recipe_not_found"
        code_issue = "The repository README does not link the requested stage to a Python entry point."
        if article and any(item.get("scope_kind") == "dataset_transfer" for item in article):
            from .fire_lr import transferred_fire_lr

            code, references, kind, code_issue = transferred_fire_lr(snapshot, article)
            code_issue_kind = kind or code_issue_kind

    if cross_page and article and code and code[0].get("model_name"):
        from .experiment_lr import subject_key

        model = code[0]["model_name"].rsplit("/", 1)[-1]
        if (subject_key(model) not in subject_key(article[0]["experiment_evidence"]["excerpt"])
                or code[0].get("optimizer") and code[0]["optimizer"].casefold() != optimizer.casefold()):
            article = []

    paper_values = {candidate["value"] for candidate in article}
    code_values = {candidate["value"] for candidate in code if candidate["value"] is not None}
    issues = []
    issue_codes = []
    if len(paper_values) != 1 or group and len(article) != 1:
        issue_codes.append("paper_lr_ambiguous" if paper_values or table_issue and "ambiguous" in table_issue
                           else "paper_lr_unconfirmed")
        if table_lr:
            issues.append(table_issue or "The selected table does not identify a unique LR cell.")
        elif group:
            issues.append("The PDF does not uniquely identify the selected group's initial learning rate within the stage.")
        elif paper_values:
            issues.append("The paper has multiple initial learning rates for this stage.")
        elif cross_page:
            issues.append("The PDF experiment and method-wide base LR could not be linked unambiguously "
                          "to the selected context and recipe.")
        elif paper_selector:
            issues.append("The PDF does not confirm a numerical initial LR for the selected section and experiment context.")
        else:
            issues.append("The PDF text has no explicit initial learning rate tied to this stage, so a comparison is not possible.")
    if code_issue:
        issue_codes.append(code_issue_kind)
        issues.append(code_issue)
    elif len(code_values) > 1:
        issue_codes.append("code_lr_ambiguous")
        issues.append("The selected shell commands specify conflicting initial learning rates." if shell_recipe
                      else "The documented entry point has multiple plausible initial learning rates.")
    elif not code_values or any(candidate["value"] is None for candidate in code):
        issue_codes.append("code_lr_unresolved")
        issues.append("The documented entry point has no uniquely resolvable explicit optimizer learning rate.")

    paper_value = next(iter(paper_values)) if len(paper_values) == 1 and (not group or len(article) == 1) else None
    code_value = next(iter(code_values)) if not code_issue and len(code_values) == 1 and all(
        candidate["value"] is not None for candidate in code
    ) else None
    if issues:
        status = "insufficient_materials"
        explanation = " ".join(issues)
    elif paper_value == code_value:
        status = "match"
        explanation = "The paper and the documented recipe specify the same initial learning rate."
    else:
        status = "potential_discrepancy"
        explanation = "The paper and the documented recipe specify different initial learning rates. Expert confirmation required."
    if selected_script and status != "insufficient_materials":
        explanation = explanation.replace("documented recipe", "user-selected script recipe")
    if (cross_page or any(item.get("rate_kind") == "base_before_warmup" for item in code)) and status != "insufficient_materials":
        explanation = explanation.replace("initial learning rate", "base optimizer learning rate")
    return {
        "status": status, "stage": stage, "commit": snapshot.commit,
        **({"group": group} if group else {}),
        "paper_value": paper_value, "code_value": code_value,
        "evidence": {"paper": article or partial_paper, "stage_links": references, "code": code},
        "issue_codes": issue_codes,
        "explanation": explanation,
        "scope": ("Comparison with a user-selected script and generated input at this commit. "
                  "The authors' historical run is unverified." if selected_script else
                  "Comparison with the published code recipe at this commit. The authors' historical run is unverified."),
        "expert_confirmation_required": status == "potential_discrepancy",
    }
