"""Compare an explicitly staged paper claim with a documented Git entry point."""

import ast
import re
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

import pymupdf

STAGE = re.compile(r"\b(?:step|stage|phase)\s+\d+\b", re.I)
LAUNCH = re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+(?:\./)?([\w./-]+\.py)(?![\w./-])")
NUMBER = r"\+?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
PAPER_LR = re.compile(
    r"\b(?:initial\s+(?:learning rate|lr)\s*(?:of|is|was|=|:)\s*"
    r"|(?:learning rate|lr)\s+(?:(?:is|was)\s+)?initially\s+(?:set to\s+)?"
    r"|(?:learning rate|lr)\s+starts?\s+at\s+)"
    rf"({NUMBER})(?![\w+%/]|\.\d)",
    re.I,
)
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


def paper_candidates(pdf, target):
    candidates = []
    with pymupdf.open(pdf) as document:
        if not document.is_pdf or document.needs_pass:
            raise ValueError("article must be a readable, unencrypted PDF")
        for page in document:
            # Preserve text blocks so that nearby columns do not interleave.
            for block in page.get_text("blocks"):
                if block[6] != 0:
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", compact(block[4])):
                    if not target.search(sentence):
                        continue
                    # A sentence may state values for several numbered stages.
                    labels = list(STAGE.finditer(sentence))
                    parts = (
                        [sentence[label.start(): labels[i + 1].start() if i + 1 < len(labels) else len(sentence)]
                         for i, label in enumerate(labels)]
                        if len(labels) > 1 else [sentence]
                    )
                    for part in parts:
                        if describes_stage(part, target):
                            for match in PAPER_LR.finditer(part):
                                value = normalized(match[1])
                                if value is not None:
                                    candidates.append({"value": value, "page": page.number + 1, "excerpt": part})
    return candidates


def entry_points(snapshot, target):
    references = []
    for path in sorted(snapshot.files):
        if "/" in path or PurePosixPath(path).stem.lower() != "readme":
            continue
        heading = ""
        for number, line in enumerate(snapshot.read(path).splitlines(), 1):
            if re.match(r"^\s*#{1,6}\s", line):
                heading = line
            context = line if (target.search(line) or STAGE.search(line)) else heading
            if describes_stage(context, target):
                for launch in LAUNCH.finditer(line):
                    references.append({"path": path, "line_start": number, "line_end": number,
                                       "excerpt": line.strip(), "entry_point": launch[1]})
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


def audit(pdf, repo, commit, stage):
    snapshot = Snapshot(repo, commit)
    target = stage_pattern(stage)
    article = paper_candidates(pdf, target)
    references = entry_points(snapshot, target)
    paths = sorted({ref["entry_point"] for ref in references})
    code = []
    code_issue = None
    if any(re.search(r"\.py\s+-{1,2}[\w-]", ref["excerpt"]) for ref in references):
        code_issue = "The documented launch uses command-line overrides that this check cannot resolve."
    elif len(paths) == 1 and paths[0] in snapshot.files:
        try:
            code = code_candidates(snapshot, paths[0])
        except SyntaxError:
            code_issue = "The documented entry point or its configuration could not be parsed as Python."
    elif len(paths) > 1:
        code_issue = "The requested stage has multiple documented Python entry points."
    elif paths:
        code_issue = "The documented Python entry point is absent from the selected commit."
    else:
        code_issue = "The repository README does not link the requested stage to a Python entry point."

    paper_values = {candidate["value"] for candidate in article}
    code_values = {candidate["value"] for candidate in code if candidate["value"] is not None}
    issues = []
    if len(paper_values) != 1:
        issues.append("The paper has multiple initial learning rates for this stage." if paper_values
                      else "The PDF text has no explicit initial learning rate tied to this stage.")
    if code_issue:
        issues.append(code_issue)
    elif len(code_values) > 1:
        issues.append("The documented entry point has multiple plausible initial learning rates.")
    elif not code_values or any(candidate["value"] is None for candidate in code):
        issues.append("The documented entry point has no uniquely resolvable explicit optimizer learning rate.")

    paper_value = next(iter(paper_values)) if len(paper_values) == 1 else None
    code_value = next(iter(code_values)) if len(code_values) == 1 and all(
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
    return {
        "status": status, "stage": stage, "commit": snapshot.commit,
        "paper_value": paper_value, "code_value": code_value,
        "evidence": {"paper": article, "stage_links": references, "code": code},
        "explanation": explanation,
        "scope": "Comparison with the published code recipe at this commit. The authors' historical run is unverified.",
        "expert_confirmation_required": status == "potential_discrepancy",
    }
