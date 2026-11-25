"""Trace a documented test-domain rotation through generated and loaded paths."""

import ast
import re
import shlex
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from .initial_lr import Snapshot, compact, evidence, open_article, straight_line_calls

UNKNOWN = object()
TEST = re.compile(r"\b(?:test|target) domains?\b", re.I)
DEGREES = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(?:[°◦]|degrees?\b)")
ANGLE_LIST = re.compile(rf"(?:\s+of)?\s*{DEGREES.pattern}(?:(?:\s*(?:,|and|&)\s*){DEGREES.pattern})*")
COMMAND = re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+(?:\./)?([\w./-]+\.py)(.*)")


def key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def angle_text(value):
    return format(Decimal(str(value)).normalize(), "f")


def paper_findings(pdf, dataset, experiment, domain):
    matches, partial = [], []
    with open_article(pdf) as doc:
        for page in doc:
            blocks = [compact(b[4]) for b in page.get_text("blocks") if b[6] == 0]
            for index, block in enumerate(blocks):
                if key(dataset) not in key(block):
                    continue
                # Caption and corresponding result table share the same page.
                caption = re.match(r"Table\s+(\d+)\.", block, re.I)
                table = None
                if caption:
                    for following in blocks[index + 1:index + 4]:
                        if re.match(r"Table\s+\d+\.", following, re.I):
                            break
                        if re.search(r"\bDataset\b", following) and key(experiment) in key(following):
                            table = following
                            break
                for sentence in re.split(r"(?<=[.!?])\s+", block):
                    test = TEST.search(sentence)
                    named = re.search(
                        r"\b(?:test|target) domain\s+([\w.-]+)\s+(?:uses|with|has)\s+(?:a\s+)?(?:rotation\s+of\s+)?",
                        sentence, re.I,
                    )
                    if not DEGREES.search(sentence):
                        continue
                    found = {"page": page.number + 1, "excerpt": sentence}
                    if not test or (key(experiment) not in key(block) and table is None):
                        partial.append(found)
                        continue
                    if named:
                        angle = DEGREES.match(sentence, named.end())
                        values = [angle[1]] if angle and named[1].casefold() == domain.casefold() else []
                    else:
                        listed = ANGLE_LIST.match(sentence, test.end())
                        try:
                            values = [a for a in DEGREES.findall(listed[0]) if angle_text(a) == angle_text(domain)] if listed else []
                        except ArithmeticError:
                            values = []
                    if not values:
                        continue
                    if table:
                        found["experiment_context"] = table[:450]
                    elif key(experiment) not in key(sentence):
                        found["experiment_context"] = block[:500]
                    matches.extend({**found, "value": angle_text(value)} for value in values)
    return matches, partial


@dataclass(frozen=True)
class Value:
    data: object = UNKNOWN
    trace: tuple = ()


@dataclass(frozen=True)
class Rotation:
    angle: str


@dataclass(frozen=True)
class Loaded:
    path: str


@dataclass(frozen=True)
class Conversion:
    pass


@dataclass(frozen=True)
class CallableSource:
    path: str
    node: object


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else ""
    return ""


def join_trace(*traces):
    out = {}
    for trace in traces:
        for item in trace:
            out.setdefault((item["path"], item["line_start"], item["line_end"]), item)
    return tuple(out.values())


def combine(data, *values):
    return Value(data, join_trace(*(value.trace for value in values)))


class StaticTrace:
    """Interpret only whitelisted AST forms. No project modules are imported."""

    def __init__(self, snapshot, domain):
        self.snapshot = snapshot
        self.domain = domain
        self.documents = {}
        self.reads = []
        self.writes = []
        self.partial = []
        self.depth = 0

    def document(self, path):
        if path not in self.documents:
            source = self.snapshot.read(path)
            self.documents[path] = source, ast.parse(source)
        return self.documents[path]

    def location(self, path, node):
        return evidence(path, self.document(path)[0], node)

    def source_import(self, path, node, env):
        parts = path.split("/")[:-1]
        prefix = parts[:len(parts) - node.level + 1] if node.level else []
        module = "/".join([*prefix, *(node.module or "").split(".")]).strip("/")
        file = module + ".py"
        if file not in self.snapshot.files:
            return
        body = self.document(file)[1].body
        definitions = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for alias in node.names:
            if alias.name == "*":
                exports = next((n.value for n in body if isinstance(n, ast.Assign)
                                and any(dotted(t) == "__all__" for t in n.targets)), None)
                if exports is not None and not isinstance(exports, (ast.List, ast.Tuple)):
                    continue
                names = ({n.value for n in exports.elts if isinstance(n, ast.Constant)}
                         if exports is not None else {n.name for n in definitions if not n.name.startswith("_")})
            else:
                names = {alias.name}
            for name in names:
                matches = [n for n in definitions if n.name == name]
                if len(matches) == 1:
                    env[alias.asname or name] = Value(CallableSource(file, matches[0]), (self.location(path, node),))

    def expression(self, node, env, path, context=(), uncertain=False):
        def visit(child):
            return self.expression(child, env, path, context, uncertain)

        if isinstance(node, ast.Constant):
            return Value(node.value)
        if isinstance(node, (ast.Name, ast.Attribute)):
            return env.get(dotted(node), Value())
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [visit(n) for n in node.elts]
            return combine(tuple(n.data for n in items), *items)
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.Not) and value.data is not UNKNOWN:
                return combine(not value.data, value)
            if isinstance(node.op, (ast.USub, ast.UAdd)) and type(value.data) in (int, float):
                return combine(-value.data if isinstance(node.op, ast.USub) else value.data, value)
        if isinstance(node, ast.BoolOp):
            values = [visit(n) for n in node.values]
            states = [bool(n.data) for n in values if n.data is not UNKNOWN]
            if isinstance(node.op, ast.And):
                data = False if False in states else True if len(states) == len(values) else UNKNOWN
            else:
                data = True if True in states else False if len(states) == len(values) else UNKNOWN
            return combine(data, *values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = visit(node.left), visit(node.comparators[0])
            op = node.ops[0]
            if a.data is not UNKNOWN and b.data is not UNKNOWN:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    return combine((a.data == b.data) == isinstance(op, ast.Eq), a, b)
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(b.data, (tuple, str)):
                    if not isinstance(b.data, str) or isinstance(a.data, str):
                        return combine((a.data in b.data) == isinstance(op, ast.In), a, b)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = visit(node.left), visit(node.right)
            if type(a.data) is type(b.data) and isinstance(a.data, (str, int, tuple)):
                return combine(a.data + b.data, a, b)
        if isinstance(node, ast.JoinedStr):
            parts = [visit(n.value) if isinstance(n, ast.FormattedValue) and n.conversion == -1
                     and n.format_spec is None else visit(n) for n in node.values]
            if all(isinstance(part.data, (str, int, float)) for part in parts):
                return combine("".join(str(part.data) for part in parts), *parts)
        if isinstance(node, ast.Call):
            return self.call(node, env, path, context, uncertain)
        return Value()

    def call(self, node, env, path, context, uncertain):
        name = dotted(node.func)
        short = name.rsplit(".", 1)[-1]
        values = [self.expression(a, env, path, context, uncertain) for a in node.args]
        keywords = {k.arg: self.expression(k.value, env, path, context, uncertain) for k in node.keywords}
        data = [v.data for v in values]
        if name in {"str", "int", "float", "range"} and len(data) == 1:
            try:
                if type(data[0]) in (str, int, float):
                    result = {"str": str, "int": int, "float": float, "range": range}[name](data[0])
                    return combine(result, *values)
            except (TypeError, ValueError, OverflowError):
                pass
        if short == "Compose" and isinstance(node.args[0] if node.args else None, ast.List):
            transforms = node.args[0].elts
            if all(isinstance(x, ast.Call) and dotted(x.func).rsplit(".", 1)[-1] in {"ToTensor", "Normalize"}
                   for x in transforms):
                return Value(Conversion())
        if short == "rotate" and (name.startswith("transforms.") or name.startswith("torchvision.")):
            angle = values[1] if len(values) == 2 else keywords.get("angle", Value())
            if type(angle.data) in (int, float):
                location = self.location(path, node)
                rotated = Value(Rotation(angle_text(angle.data)), join_trace(context, angle.trace, (location,)))
                self.partial.append({"value": rotated.data.angle, "use": location, "definitions": list(rotated.trace[:-1])})
                return rotated
        if name == "torch.save" and len(values) == 2 and isinstance(data[1], str):
            angle = data[0].angle if isinstance(data[0], Rotation) else None
            self.writes.append((data[1], angle, uncertain,
                                join_trace(context, values[0].trace, values[1].trace, (self.location(path, node),))))
        if name == "torch.load" and len(values) == 1 and isinstance(data[0], str):
            return Value(Loaded(data[0]), join_trace(context, values[0].trace, (self.location(path, node),)))
        receiver = name.rpartition(".")[0].rsplit(".", 1)[-1]
        if (short == "append" and len(values) == 1 and isinstance(data[0], Loaded)
                and re.search(r"(?:^|_)(?:images?|imgs?|pixels?)(?:_|$)", receiver, re.I)):
            self.use_read(values[0], path, node, context, uncertain)
        target = env.get(name, Value()).data
        if isinstance(target, Conversion) and len(data) == 1 and isinstance(data[0], Rotation):
            return values[0]
        if isinstance(target, CallableSource) and None not in keywords and self.depth < 6:
            site = self.location(path, node)
            self.invoke(target, values, keywords, env, join_trace(context, env[name].trace, (site,)), uncertain)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "__init__"
                and isinstance(node.func.value, ast.Call) and dotted(node.func.value.func) == "super"):
            target = env.get("super.__init__", Value()).data
            if isinstance(target, CallableSource) and self.depth < 6:
                self.invoke(target, values, keywords, env, join_trace(context, (self.location(path, node),)), uncertain)
        return Value()

    def use_read(self, value, path, node, context, uncertain):
        self.reads.append((value.data.path, uncertain,
                           join_trace(context, value.trace, (self.location(path, node),))))

    def invoke(self, target, values, keywords, parent, context, uncertain):
        self.depth += 1
        try:
            path, node = target.path, target.node
            env = {k: v for k, v in parent.items() if k.startswith(("args.", "self.", "super."))}
            self.bind_module(path, env)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        env["self." + method.name] = Value(CallableSource(path, method))
                if len(node.bases) == 1:
                    base = env.get(dotted(node.bases[0]), Value()).data
                    if isinstance(base, CallableSource) and isinstance(base.node, ast.ClassDef):
                        initializers = [f for f in base.node.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"]
                        if len(initializers) == 1:
                            env["super.__init__"] = Value(CallableSource(base.path, initializers[0]))
                init = env.get("self.__init__", Value()).data
                if isinstance(init, CallableSource):
                    self.invoke(init, values, keywords, env, context, uncertain)
                return
            params = [a.arg for a in node.args.args]
            if params and params[0] == "self":
                params.pop(0)
            if len(values) > len(params) or any(k not in params for k in keywords):
                return
            for param, default in zip(params[len(params) - len(node.args.defaults):], node.args.defaults):
                env[param] = self.expression(default, env, path)
            env.update(zip(params, values))
            env.update(keywords)
            self.walk(node.body, env, path, context, uncertain)
            parent.update((k, v) for k, v in env.items() if k.startswith("self.") and not isinstance(v.data, CallableSource))
        finally:
            self.depth -= 1

    def bind_module(self, path, env):
        for node in self.document(path)[1].body:
            if isinstance(node, ast.FunctionDef):
                env[node.name] = Value(CallableSource(path, node))
            elif isinstance(node, ast.ImportFrom):
                self.source_import(path, node, env)

    def walk(self, body, env, path, context=(), uncertain=False):
        for node in body:
            if isinstance(node, ast.Assign):
                value = self.expression(node.value, env, path, context, uncertain)
                for target in node.targets:
                    name = dotted(target)
                    if isinstance(target, ast.Subscript):
                        name = dotted(target.value)
                        # A rotated per-image value is the provenance of the saved buffer.
                        if not isinstance(value.data, Rotation):
                            if isinstance(env.get(name, Value()).data, Rotation):
                                env[name] = Value()
                            continue
                    if name:
                        site = (self.location(path, node),)
                        source_name = dotted(node.value)
                        attributes = [(key[len(source_name):], v) for key, v in env.items()
                                      if source_name and key.startswith(source_name + ".")]
                        env[name] = Value(value.data, join_trace(value.trace, site)) if value.data is not UNKNOWN else value
                        for suffix, attribute in attributes:
                            env[name + suffix] = Value(attribute.data, join_trace(attribute.trace, site))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.call(node.value, env, path, context, uncertain)
            elif isinstance(node, ast.ImportFrom):
                self.source_import(path, node, env)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test, env, path)
                trail = join_trace(context, condition.trace, (self.location(path, node.test),))
                if condition.data is UNKNOWN:
                    yes, no = env.copy(), env.copy()
                    stop_yes = self.walk(node.body, yes, path, trail, True)
                    stop_no = self.walk(node.orelse, no, path, trail, True)
                    if stop_yes and stop_yes == stop_no:
                        return stop_yes
                    uncertain |= bool(stop_yes or stop_no)
                    for name in yes.keys() | no.keys():
                        a, b = yes.get(name, Value()), no.get(name, Value())
                        env[name] = combine(a.data, a, b) if a.data == b.data else Value()
                else:
                    stop = self.walk(node.body if condition.data else node.orelse, env, path, trail, uncertain)
                    if stop:
                        return stop
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                collection = self.expression(node.iter, env, path)
                data = collection.data
                if isinstance(data, (tuple, range)) and len(data) <= 64:
                    items = data
                    label = node.target.id.lower()
                    if "domain" in label:
                        items = [item for item in items if str(item) == self.domain]
                    for item in items:
                        env[node.target.id] = combine(item, collection)
                        stop = self.walk(node.body, env, path, context, uncertain)
                        if stop == "return":
                            return stop
                        if stop == "break":
                            break
                elif data is UNKNOWN:
                    env[node.target.id] = Value()
                    # Only buffer provenance survives a symbolic per-image loop.
                    branch = env.copy()
                    self.walk(node.body, branch, path, context, True)
                    for name, value in branch.items():
                        if isinstance(value.data, Rotation) and any(
                            isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and dotted(n.value) == name
                            for n in ast.walk(node)
                        ):
                            env[name] = value
                        elif name in env and env[name] != value:
                            env[name] = Value()
            elif isinstance(node, ast.Return):
                value = self.expression(node.value, env, path, context, uncertain)
                if isinstance(value.data, Loaded):
                    self.use_read(value, path, node, context, uncertain)
                return "return"
            elif isinstance(node, ast.Continue):
                return "continue"
            elif isinstance(node, ast.Break):
                return "break"
        return None


def documented_commands(snapshot, dataset, experiment):
    result = []
    for path in sorted(snapshot.files):
        if "/" in path or not re.fullmatch(r"README(?:\.[\w-]+)?", path, re.I):
            continue
        lines = snapshot.read(path).splitlines()
        context = ""
        for index, line in enumerate(lines):
            match = COMMAND.search(line)
            if not match:
                if key(dataset) in key(line) or key(experiment) in key(line):
                    context = line.strip()
                continue
            script, tail = match.groups()
            if script not in snapshot.files:
                continue
            try:
                args = shlex.split(tail.strip().strip(chr(96) + "*"), comments=True)
            except ValueError:
                continue
            options = {}
            for token in args:
                if token.startswith("--"):
                    option, sep, value = token.partition("=")
                    if option in options:
                        break
                    options[option] = [value] if sep else []
                elif options:
                    options[next(reversed(options))].append(token)
                else:
                    break
            else:
                if options and all(options.values()) and (key(dataset) in key(context) or key(dataset) in key(line)):
                    result.append((script, options, {
                        "path": path, "line_start": index + 1, "line_end": index + 1, "excerpt": line.strip(),
                        "context": context,
                    }))
    return result


def command_environment(tracer, script, options):
    env = {"__name__": Value("__main__")}
    tracer.bind_module(script, env)
    declarations = {}
    parser_name = None
    for node in tracer.document(script)[1].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call):
            if dotted(node.value.func).endswith(".parse_args"):
                parser_name = dotted(node.value.func).rsplit(".", 1)[0]
                destination = dotted(node.targets[0])
    if not parser_name:
        return None
    for call in straight_line_calls(tracer.document(script)[1].body):
        if dotted(call.func) != f"{parser_name}.add_argument" or not call.args:
            continue
        names = [arg.value for arg in call.args if isinstance(arg, ast.Constant)
                 and isinstance(arg.value, str) and arg.value.startswith("--")]
        flags = {item.arg: item.value for item in call.keywords}
        if not names or {"action", "dest", None} & flags.keys():
            continue
        for name in names:
            declarations[name] = (names[0], flags, call)
    explicit = {}
    for option, values in options.items():
        possible = [name for name in declarations if name.startswith(option)] if option not in declarations else [option]
        if len(possible) != 1:
            return None
        explicit[declarations[possible[0]][0]] = values
    for name, flags, call in declarations.values():
        if name in explicit:
            convert = {"int": int, "float": float, "str": str}.get(dotted(flags.get("type")), str)
            try:
                args = tuple(convert(v) for v in explicit[name])
            except ValueError:
                return None
            data = args if "nargs" in flags else args[0] if len(args) == 1 else UNKNOWN
        else:
            data = tracer.expression(flags.get("default"), env, script).data
        if data is not UNKNOWN:
            env[destination + "." + name[2:].replace("-", "_")] = Value(data, (tracer.location(script, call),))
    return env


def code_findings(snapshot, dataset, experiment, domain):
    commands = documented_commands(snapshot, dataset, experiment)
    links = []
    writes, reads, partial = [], [], []
    for script, options, ref in commands:
        flattened = [v for values in options.values() for v in values]
        experiment_named = key(experiment) in key(ref["context"]) or any(
            key(v).startswith(key(experiment)) for v in flattened
        )
        metrics = [key(values[0]) for flag, values in options.items() if "metric" in flag]
        if metrics and not all(metric in {"acc", "accuracy"} for metric in metrics):
            continue
        documents_test = metrics or any(
            key(flag) in {"split", "datacase"} and values == ["test"] for flag, values in options.items()
        ) or re.search(r"\b(?:test|evaluate|evaluation)\b", ref["context"], re.I)
        mode = "test" if experiment_named and documents_test else "generate"
        tracer = StaticTrace(snapshot, domain)
        try:
            env = command_environment(tracer, script, options)
            if env is None:
                continue
            tracer.walk(tracer.document(script)[1].body, env, script, (ref,))
        except SyntaxError:
            continue
        # Both ends must include the same documented dataset key.
        def dataset_path(name):
            return any(part == value or part.startswith(value + "_") for part in name.split("/")
                       for flag, values in options.items() if "dataset" in flag for value in values)

        if mode == "generate":
            writes.extend(item for item in tracer.writes if dataset_path(item[0]))
            partial.extend(tracer.partial)
        else:
            reads.extend(item for item in tracer.reads if dataset_path(item[0]))
        if (tracer.reads if mode == "test" else tracer.writes or tracer.partial):
            links.append(ref)

    def test_domain(path):
        return ("test" in path.split("/") and bool(re.search(
            rf"(?<![\w+-]){re.escape(domain)}(?![A-Za-z0-9.+-])",
            path.replace("_", "/").rsplit(".", 1)[0],
        )))

    candidates = []
    test_reads = [item for item in reads if test_domain(item[0])]
    for filepath, angle, write_uncertain, write_trace in writes:
        if angle is None or not test_domain(filepath):
            continue
        for read_path, read_uncertain, read_trace in test_reads:
            if filepath == read_path and not write_uncertain and not read_uncertain:
                candidates.append({"value": angle, "use": write_trace[-1], "artifact": filepath,
                                   "definitions": list(write_trace[:-1]), "stage_trace": list(read_trace)})
    # Keep conflicting writers and unsupported route failures visible.
    read_paths = {path for path, _, _ in test_reads}
    unresolved_write = any(
        p in read_paths and (uncertain or angle is None) for p, angle, uncertain, _ in writes
    )
    unlinked_read = read_paths - {row["artifact"] for row in candidates}
    unresolved_read = any(uncertain for _, uncertain, _ in test_reads)
    if candidates and not (unresolved_write or unresolved_read or unlinked_read):
        return unique(candidates), links, None
    if candidates:
        reason = "At least one test artifact has an unresolved rotation, producer or loader."
    elif not test_reads:
        reason = "Could not trace a documented evaluation to an image file for this test domain."
    elif not any(test_domain(path) for path, angle, _, _ in writes if angle is not None):
        reason = "The traced rotation is not saved for this test domain."
    else:
        reason = "The rotated test artifact could not be linked to the file read by the test loader."
    read_evidence = [{"artifact": p, "use": t[-1], "stage_trace": list(t[:-1])}
                     for p, _, t in test_reads]
    available_writes = [{"artifact": p, "value": angle, "use": t[-1], "definitions": list(t[:-1])}
                        for p, angle, _, t in writes if angle is not None]
    return unique([*candidates, *available_writes, *partial, *read_evidence]), links, reason


def unique(findings):
    unique_rows = {}
    for finding in findings:
        unique_rows.setdefault((finding.get("artifact"), finding.get("value"),
                                finding["use"]["path"], finding["use"]["line_start"]), finding)
    return list(unique_rows.values())


def audit_rotation(pdf, repo, commit, experiment, dataset, domain, *, memory=None):
    snapshot = Snapshot(repo, commit)
    if memory:
        from .rotation_memory import memory_findings, variant_paper

        extra, table, producer, evaluator, field = memory
        domain, extra = angle_text(domain), angle_text(extra)
        paper, partial = variant_paper(pdf, dataset, domain, extra, table)
        code, links, issue = memory_findings(snapshot, dataset, domain, extra, producer, evaluator, field)
    else:
        paper, partial = paper_findings(pdf, dataset, experiment, domain)
        code, links, issue = code_findings(snapshot, dataset, experiment, domain)
    paper_values = {item["value"] for item in paper}
    code_values = {item["value"] for item in code if "value" in item} if not issue else set()
    paper_value = next(iter(paper_values)) if len(paper_values) == 1 else None
    code_value = next(iter(code_values)) if len(code_values) == 1 else None
    reasons = []
    if paper_value is None:
        reasons.append("The paper does not give a unique rotation for this experiment's test domain.")
    if issue:
        reasons.append(issue)
    elif code_value is None:
        reasons.append("The test loader can receive artifacts with different rotation angles.")
    if reasons:
        status = "insufficient_materials"
    else:
        status = "match" if paper_value == code_value else "potential_discrepancy"
        reasons.append("The paper and the documented test recipe specify the same rotation angle."
                       if status == "match" else
                       "The paper and the documented test recipe specify different rotation angles. Expert confirmation required.")
    if memory:
        reasons = [reason.replace("documented test recipe", "user-selected in-memory test recipe") for reason in reasons]
    return {
        "status": status, "stage": experiment, "check": "test-rotation", "commit": snapshot.commit,
        "dataset": dataset, "domain": domain, "split": "test", "unit": "degrees",
        "paper_value": paper_value, "code_value": code_value,
        "evidence": {"paper": paper or partial, "stage_links": links, "code": code},
        "explanation": " ".join(reasons),
        "scope": ("Static comparison of a user-selected configuration recipe at this commit."
                  if memory else "Static comparison of the documented test recipe at this commit."),
        "expert_confirmation_required": status == "potential_discrepancy",
    }
