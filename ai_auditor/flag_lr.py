"""Read a documented experiment selector and a TensorFlow flag-backed SGD rate."""

import ast
import re
import shlex

from .initial_lr import assignments, evidence, literal, module_paths, name_writes, straight_line_calls


def launch_options(tail, *, allow_short=False):
    # Accept literal flag/value pairs only. Shell substitutions and extra
    # commands cannot be used as evidence for a default.
    segment = re.split(r"[`*]", tail, maxsplit=1)[0]
    try:
        args = shlex.split(segment, comments=True)
    except ValueError:
        return None
    if not args:
        return None
    options = {}
    flag_pattern = r"(?:--[A-Za-z_]\w*(?:-\w+)*|-[A-Za-z])" if allow_short else r"--[A-Za-z_]\w*(?:-\w+)*"
    value_pattern = r"[A-Za-z0-9_./+-]+" if allow_short else r"[A-Za-z0-9_.+-]+"
    cursor = iter(args)
    for token in cursor:
        option, sep, value = token.partition("=")
        if not sep:
            value = next(cursor, "")
        if (not re.fullmatch(flag_pattern, option) or option in options
                or not re.fullmatch(value_pattern, value)):
            return None
        options[option] = value
    return options


def selector_matches(stage, options):
    def words(text):
        return re.findall(r"[^\W_]+", text.casefold())
    return [option for option, value in (options or {}).items()
            if re.search(r"[A-Za-z]", value) and words(stage) == words(value)]


def chain(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*chain(node.value), node.attr)
    return ()


class Sources:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.cache = {}

    def module(self, name):
        paths = module_paths(self.snapshot, name, entry=False)
        return paths[0] if len(paths) == 1 else None

    def read(self, path):
        if path not in self.cache:
            source = self.snapshot.read(path)
            self.cache[path] = (source, ast.parse(source).body)
        return self.cache[path]

    def site(self, path, node):
        return evidence(path, self.read(path)[0], node)

    def explicit_tensorflow(self, path, alias):
        return [node for node in self.read(path)[1] if isinstance(node, ast.Import)
                and any(item.name == "tensorflow" and (item.asname or item.name) == alias for item in node.names)]

    def tensorflow(self, path, alias):
        body = self.read(path)[1]
        if name_writes(body, alias):
            return []
        direct = [self.site(path, n) for n in self.explicit_tensorflow(path, alias)]
        if direct:
            return direct
        # A wildcard import can carry a public TensorFlow alias from a local
        # module. Resolve the exports statically, without importing it.
        proof = []
        for node in body:
            if not (isinstance(node, ast.ImportFrom) and not node.level and node.module
                    and any(item.name == "*" for item in node.names)):
                continue
            imported = self.module(node.module)
            if not imported:
                return []
            external = self.read(imported)[1]
            if name_writes(external, alias) or name_writes(external, "__all__"):
                return []
            matches = self.explicit_tensorflow(imported, alias)
            if matches:
                proof.extend([self.site(path, node), *(self.site(imported, match) for match in matches)])
        return proof

    def aliases(self, path):
        _, body = self.read(path)
        registries = {}
        namespaces = {}
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            name = statement.targets[0].id
            if name_writes(body, name) != 1:
                continue
            value = chain(statement.value)
            if value[1:] == ("app", "flags"):
                imported = self.tensorflow(path, value[0])
                if imported:
                    registries[name] = [*imported, self.site(path, statement)]
            if len(value) == 2 and value[0] in registries and value[1] == "FLAGS":
                namespaces[name] = registries[value[0]] + [self.site(path, statement)]
        return registries, namespaces

    def flag_declarations(self, path):
        _, body = self.read(path)
        registries, _ = self.aliases(path)
        results = {}
        for call in straight_line_calls(body):
            name = chain(call.func)
            if (len(name) == 2 and name[0] in registries and name[1] in {"DEFINE_string", "DEFINE_float"}
                    and len(call.args) == 3 and not call.keywords
                    and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                results.setdefault(call.args[0].value, []).append((name[1], call))
        return results

    def constructors(self, path):
        _, body = self.read(path)
        imports = {}
        for node in body:
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                model_path = self.module(node.module)
                if model_path:
                    for item in node.names:
                        imports.setdefault(item.asname or item.name, []).append((model_path, item.name, node))
        for call in straight_line_calls(body):
            name = getattr(call.func, "id", "")
            aliases = assignments(body, name)
            if len(aliases) == 1 and name_writes(body, name) == 1 and aliases[0].lineno < call.lineno:
                name = getattr(aliases[0].value, "id", "")
            options = imports.get(name, [])
            if len(options) != 1 or name_writes(body, name):
                continue
            location, class_name, imported = options[0]
            matches = [node for node in self.read(location)[1]
                       if isinstance(node, ast.ClassDef) and node.name == class_name]
            if len(matches) == 1:
                trail = [self.site(path, imported), *(self.site(path, alias) for alias in aliases),
                         self.site(path, call)]
                yield location, matches[0], trail


def flag_mutated(body, namespaces, name):
    nodes = [node for statement in body for node in ast.walk(statement)]
    aliases = {(alias,) for alias in namespaces}
    for node in nodes:
        if isinstance(node, ast.Assign) and chain(node.value) in aliases:
            aliases.update(chain(target) for target in node.targets if chain(target))
    for node in nodes:
        if isinstance(node, ast.Attribute) and chain(node)[:-1] in aliases and node.attr == name:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                return True
        if isinstance(node, ast.Subscript) and chain(node.value) in aliases:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                return True
        if isinstance(node, ast.Call):
            if any(chain(arg) in aliases for arg in [*node.args, *(kw.value for kw in node.keywords)]):
                return True
            if chain(node.func)[:-1] in aliases:
                return True
    return False


def flag_lr(snapshot, reference):
    path = reference["entry_point"]
    options = reference.get("arguments")
    selectors = reference.get("experiment_options", [])
    if not options or len(selectors) != 1:
        raise ValueError("The documented launch does not identify one literal experiment option.")
    selector = selectors[0][2:]
    sources = Sources(snapshot)
    source, body = sources.read(path)
    declarations = sources.flag_declarations(path)
    _, entry_namespace = sources.aliases(path)
    selection = declarations.get(selector, [])
    used = [call for call in straight_line_calls(body) if any(
        chain(arg) in {(alias, selector) for alias in entry_namespace} for arg in call.args
    )]
    if (len(selection) != 1 or selection[0][0] != "DEFINE_string" or not used
            or flag_mutated(body, entry_namespace, selector)):
        raise ValueError("The documented experiment option could not be tied to a declared input in the script.")
    candidates = []
    for model_path, cls, trail in sources.constructors(path):
        _, model_body = sources.read(model_path)
        _, namespace = sources.aliases(model_path)
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "__init__":
                continue
            direct = set(straight_line_calls(method.body))
            for call in ast.walk(method):
                if not isinstance(call, ast.Call):
                    continue
                name = chain(call.func)
                if name[1:] != ("train", "GradientDescentOptimizer") or not sources.tensorflow(model_path, name[0]):
                    continue
                args = [k.value for k in call.keywords if k.arg == "learning_rate"]
                ref = chain(args[0]) if len(args) == 1 else ()
                flag = ref[1] if len(ref) == 2 and ref[0] in namespace else None
                declared = declarations.get(flag, [])
                if flag and (flag_mutated(body, entry_namespace, flag)
                             or flag_mutated(model_body, namespace, flag)):
                    raise ValueError("The learning-rate flag can be modified by the training code.")
                if (call not in direct or len(declared) != 1 or declared[0][0] != "DEFINE_float"
                        or any(sources.flag_declarations(p).get(flag) for p in list(sources.cache) if p != path)):
                    candidates.append({"value": None, "use": sources.site(model_path, call), "definitions": []})
                    continue
                declaration = declared[0][1]
                if any(option[2:].replace("-", "_") == flag for option in options):
                    raise ValueError("The documented launch overrides the learning-rate flag.")
                if len(options) != 1:
                    raise ValueError("The documented launch has additional unresolved overrides.")
                candidates.append({
                    "value": literal(source, declaration.args[1]),
                    "use": sources.site(model_path, call),
                    "definitions": [sources.site(path, declaration)],
                    "stage_trace": [sources.site(path, selection[0][1]), *(sources.site(path, item) for item in used),
                                    *trail, *namespace[ref[0]]],
                })
    if not candidates or any(item["value"] is None for item in candidates):
        raise ValueError("Could not uniquely trace the published recipe's LR flag to a TensorFlow SGD call.")
    return candidates
