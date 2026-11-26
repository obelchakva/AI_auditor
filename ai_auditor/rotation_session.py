"""Trace a README recipe's in-memory, batched TensorFlow evaluation."""

import ast
import re
from pathlib import PurePosixPath

from .experiment_lr import declared_options
from .initial_lr import main_body, name_writes, open_article, straight_line_calls
from .paper_scope import text_blocks
from .rotation import UNKNOWN, Value, angle_text, dotted, key
from .rotation_memory import MemoryTrace, assigned, call_is, require_unique


def study_method(pdf):
    with open_article(pdf) as article:
        names = {match[1] for item in text_blocks(article) if item["page"] == 1
                 for match in re.finditer(r"\b(?:we present|we propose|we introduce)\s+([A-Z][A-Z0-9]+)\s*\(",
                                          item["excerpt"], re.I)
                 if match[1].isupper()}
    return next(iter(names)) if len(names) == 1 else None


def recipes(snapshot, dataset, stage, method):
    found = []
    for path in sorted(snapshot.files):
        if "/" in path or PurePosixPath(path).stem.casefold() != "readme":
            continue
        active, method_label = False, None
        for number, line in enumerate(snapshot.read(path).splitlines(), 1):
            heading = re.match(r"^(?:\*+|#+)\s+(.+)", line)
            if heading:
                active, method_label = key(heading[1]) == key(stage), None
                continue
            if not active:
                continue
            label = re.fullmatch(r"\s*([\w-]+):\s*", line)
            if label:
                method_label = label[1]
                continue
            launch = re.fullmatch(r"\s*`?python(?:3)?\s+([\w./-]+\.py)\s+([^`]+)`?\s*", line)
            if not launch or method and method_label != method:
                continue
            from .flag_lr import launch_options

            options = launch_options(launch[2], allow_short=True)
            if not options or not options.get("--dataset") or not key(dataset).endswith(key(options["--dataset"])):
                continue
            paths = [p for p in snapshot.files if p == launch[1] or p.endswith("/" + launch[1])]
            # A same-named script is accepted only if it declares all the
            # documented options and has a guarded entry point.
            for script in sorted(paths):
                found.append({"path": path, "line_start": number, "line_end": number,
                              "excerpt": line.strip(), "entry_point": script, "method": method_label,
                              "arguments": options})
    return found


def function(body, name):
    return require_unique([node for node in body if isinstance(node, ast.FunctionDef)
                           and node.name == name and not node.decorator_list],
                          f"Cannot resolve one function {name}.")


def class_route(tracer, recipe, trail):
    path = recipe["entry_point"]
    body = tracer.document(path)[1].body
    invocation = require_unique([call for node in body for call in straight_line_calls(main_body(node))
                                 if isinstance(call.func, ast.Name) and not call.args and not call.keywords],
                                "Cannot resolve a guarded evaluation entry point.")
    entry = function(body, invocation.func.id)
    parser_call = require_unique([(target.id, call) for target, call, _ in assigned(entry.body)
                                  if isinstance(target, ast.Name) and isinstance(call, ast.Call)
                                  and isinstance(call.func, ast.Name) and not call.args and not call.keywords],
                                 "Cannot resolve evaluation argument parsing.")
    namespace, call = parser_call
    parser = function(body, call.func.id)
    _, _, values = declared_options(parser, recipe["arguments"])
    modules = {}
    for node in body:
        if isinstance(node, ast.ImportFrom) and not node.level and node.module and "." not in node.module:
            local = str(PurePosixPath(path).parent / (node.module + ".py"))
            if local in tracer.snapshot.files:
                for alias in node.names:
                    modules[alias.asname or alias.name] = (local, alias.name, node)
    candidates = []
    for node in ast.walk(entry):
        if not (isinstance(node, ast.With) and len(node.items) == 1):
            continue
        session = node.items[0]
        if not call_is(session.context_expr, "tf.Session") or not isinstance(session.optional_vars, ast.Name):
            continue
        for target, ctor, binding in assigned(node.body):
            if (not isinstance(target, ast.Name) or not isinstance(ctor, ast.Call)
                    or dotted(ctor.func) not in modules or ctor.keywords
                    or [dotted(arg) for arg in ctor.args] != [session.optional_vars.id, namespace]):
                continue
            for branch in node.body:
                if not isinstance(branch, ast.If):
                    continue
                env = {namespace + "." + name: Value(value) for name, value in values.items()}
                state = tracer.expression(branch.test, env, path).data
                if state is not True:
                    continue
                for evaluation in straight_line_calls(branch.body):
                    if (dotted(evaluation.func) != target.id + ".test" or evaluation.args
                            or len(evaluation.keywords) != 1 or evaluation.keywords[0].arg != "full"
                            or not isinstance(evaluation.keywords[0].value, ast.Constant)
                            or evaluation.keywords[0].value.value is not True):
                        continue
                    candidates.append((modules[dotted(ctor.func)], binding, branch.test, evaluation))
    (class_path, class_name, imported), binding, condition, evaluation = require_unique(
        candidates, "Cannot resolve one documented full evaluation in the selected execution branch.")
    trail.extend(tracer.location(path, node) for node in (imported, invocation, call, binding, condition, evaluation))
    cls = require_unique([node for node in tracer.document(class_path)[1].body
                          if isinstance(node, ast.ClassDef) and node.name == class_name],
                         "The evaluation class is absent or ambiguous.")
    return class_path, cls


def field_pair(node, args):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == args)


def batch_feed(tracer, path, cls, trail):
    method = function(cls.body, "test")
    if [arg.arg for arg in method.args.args] != ["self", "full"]:
        raise ValueError("The evaluator's full-mode signature is unsupported.")
    selector = require_unique([node for node in method.body if isinstance(node, ast.If) and dotted(node.test) == "full"],
                              "Cannot identify the full-test array selection.")
    options = {}
    for target, expr, assignment in assigned(selector.body):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(expr, (ast.Tuple, ast.List)):
            options.update((dotted(a), (b, assignment)) for a, b in zip(target.elts, expr.elts))
        else:
            options[dotted(target)] = (expr, assignment)
    feeds = []
    for loop in method.body:
        if not isinstance(loop, ast.For):
            continue
        for call in straight_line_calls(loop.body):
            if dotted(call.func) != "self.sess.run" or not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
                continue
            if not any(dotted(fetch) == "self.test_accuracy" for fetch in call.args[0].elts):
                continue
            feeds.extend((loop, call, kw.value) for kw in call.keywords if kw.arg == "feed_dict")
    loop, call, feed_ref = require_unique(feeds, "Cannot identify a unique batched evaluation Session.run call.")
    # Dataflow starts at the input consumed by the test graph, not just any
    # key of a feed dictionary (which also contains labels and domains).
    graph_inputs = [node.args[0] for node in ast.walk(cls) if isinstance(node, ast.Call)
                    and dotted(node.func) == "self.network" and node.args
                    and any(kw.arg == "is_training" and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False for kw in node.keywords)]
    graph_input = require_unique(graph_inputs, "Cannot identify the evaluation graph's image input.")
    feed, feed_site = require_unique([(expr, site) for dest, expr, site in assigned(loop.body)
                                     if dotted(dest) == dotted(feed_ref) and isinstance(expr, ast.Dict)
                                     and site.lineno < call.lineno], "Cannot resolve the evaluation feed dictionary.")
    pixel = require_unique([value for k, value in zip(feed.keys, feed.values) if dotted(k) == dotted(graph_input)],
                           "The evaluation feed does not supply the graph's image input.")
    batch, batch_site = require_unique([(value, site) for dest, value, site in assigned(loop.body)
                                       if dotted(dest) == dotted(pixel) and isinstance(value, ast.Subscript)
                                       and isinstance(value.slice, ast.Slice) and site.lineno < feed_site.lineno],
                                      "The evaluation does not feed slices of a traced image array.")
    selected = options.get(dotted(batch.value))
    if not selected or not field_pair(selected[0], "self"):
        raise ValueError("The fed image batches do not come from the selected full-test array.")
    if (name_writes(method.body, dotted(batch.value)) != 2 or name_writes(loop.body, dotted(pixel)) != 1
            or name_writes(loop.body, dotted(feed_ref)) != 1):
        raise ValueError("The image array or feed is reassigned before evaluation.")
    trail.extend(tracer.location(path, node) for node in (selector.test, selected[1], batch_site, feed_site, call, graph_input))
    return dotted(selected[0])


def rotated_return(fn, target):
    """Accept a direct rotation result or arithmetic with local scalar constants."""
    constants = set()
    for destination, value, _ in assigned(fn.body):
        if isinstance(destination, ast.Name) and isinstance(value, ast.Constant) and type(value.value) in (int, float):
            constants.add(destination.id)
        elif isinstance(destination, ast.Tuple) and isinstance(value, ast.Tuple):
            constants.update(a.id for a, b in zip(destination.elts, value.elts)
                             if isinstance(a, ast.Name) and isinstance(b, ast.Constant) and type(b.value) in (int, float))
    expression = fn.body[-1].value if isinstance(fn.body[-1], ast.Return) else None

    def classify(node):
        if isinstance(node, ast.Name):
            return "rotated" if node.id == target else "scalar" if node.id in constants else None
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return "scalar"
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = classify(node.left), classify(node.right)
            if left and right and right == "scalar":
                return left
        return None

    return classify(expression) == "rotated"


def rotation_wrapper(tracer, path, call, angle):
    source, tree = tracer.document(path)
    fn = function(tree.body, dotted(call.func))
    args = [arg.arg for arg in fn.args.args]
    if len(args) < 2 or len(call.args) != 2 or len(fn.body) < 2:
        raise ValueError("Cannot bind the image and angle in the rotation wrapper.")
    aliases = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "scipy.ndimage" and not node.level
               for alias in node.names if alias.name == "rotate"}
    operations = [(target, value, site) for target, value, site in assigned(fn.body)
                  if isinstance(value, ast.Call) and dotted(value.func) in aliases and len(value.args) == 2
                  and dotted(value.args[0]) == args[0]]
    target, operation, site = require_unique(operations, "Cannot locate one SciPy image rotation in the wrapper.")
    returned = fn.body[-1]
    if not rotated_return(fn, dotted(target)):
        raise ValueError("The rotation wrapper does not return the rotated image.")
    value = tracer.expression(operation.args[1], {args[1]: Value(angle)}, path).data
    if type(value) not in (int, float) or name_writes(fn.body, args[1]) or name_writes(fn.body, dotted(target)) != 1:
        raise ValueError("The actual SciPy rotation angle or output is unresolved.")
    return angle_text(value), operation, site, returned


def constructor_scope(tracer, path, init, stored, dataset, trail):
    """Verify literal constructor guards and the fresh no-cache branch."""
    parents = {child: node for node in ast.walk(init) for child in ast.iter_child_nodes(node)}
    env = {dotted(target): tracer.expression(value, {}, path) for target, value, site in assigned(init.body)
           if site.lineno < stored.lineno and dotted(target)}
    node, cache = stored, False
    while parents[node] is not init:
        parent = parents[node]
        if not isinstance(parent, ast.If):
            raise ValueError("The test-array constructor is nested in unsupported control flow.")
        value = tracer.expression(parent.test, env, path).data
        if value is UNKNOWN:
            check = parent.test
            if (node not in parent.orelse or not call_is(check, "os.path.exists")
                    or len(check.args) != 1 or check.keywords):
                raise ValueError("An unresolved constructor branch controls the full-test array.")
            filename = tracer.expression(check.args[0], env, path).data
            if filename is UNKNOWN:
                expr = next((v for t, v, _ in assigned(init.body) if dotted(t) == dotted(check.args[0])), None)
                if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod) and isinstance(expr.left, ast.Constant)
                        and isinstance(expr.left.value, str) and expr.left.value.count("%s") == 1
                        and dotted(expr.right) == "self.dataset_name"):
                    filename = expr.left.value.replace("%s", dataset)
            candidate = PurePosixPath(filename) if isinstance(filename, str) else None
            if (candidate is None or candidate.is_absolute() or ".." in candidate.parts
                    or str(PurePosixPath(path).parent / candidate) in tracer.snapshot.files
                    or str(candidate) in tracer.snapshot.files):
                raise ValueError("The selected local cache cannot be ruled out in the pinned recipe.")
            cache = True
        elif (node in parent.body) != bool(value):
            raise ValueError("The rotated array is constructed only in an inactive branch.")
        trail.append(tracer.location(path, parent.test))
        node = parent
    return cache


def image_buffer(tracer, path, cls, field, domain, dataset, trail):
    init = function(cls.body, "__init__")
    stores = [node for node in ast.walk(init) if isinstance(node, ast.Assign) and len(node.targets) == 1
              and dotted(node.targets[0]) == field and call_is(node.value, "np.array") and len(node.value.args) == 1]
    stored = require_unique(stores, "Cannot identify one array built for full-test evaluation.")
    uses_cache = constructor_scope(tracer, path, init, stored, dataset, trail)
    buffer = dotted(stored.value.args[0])
    scopes = [init.body, *(b for node in ast.walk(init) if isinstance(node, ast.If) for b in (node.body, node.orelse))]
    scope = require_unique([body for body in scopes if stored in body], "The test array branch is ambiguous.")
    # Refuse any other writes to the test field outside the selected constructor
    # branch. In particular training/evaluation must not replace it.
    if any(isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)) and dotted(n) == field
           for member in cls.body if member is not init for n in ast.walk(member)):
        raise ValueError("The full-test image field can be overwritten outside its constructor.")
    prior = scope[:scope.index(stored)]
    loops = [loop for loop in prior if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
             and any(isinstance(c, ast.Call) and dotted(c.func) == buffer + ".append" for c in ast.walk(loop))]
    angles = loops[-1] if loops else None
    if angles is None:
        raise ValueError("The full-test array has no traced rotated-image append.")
    wrapper = angles.iter
    label_list = wrapper.args[0] if call_is(wrapper, "tqdm.tqdm") and len(wrapper.args) == 1 else wrapper
    sequence = require_unique([expr for target, expr, _ in assigned(prior)
                               if dotted(target) == dotted(label_list) and isinstance(expr, (ast.List, ast.Tuple))],
                              "The named test-domain angle list is unresolved.")
    literal_angles = [tracer.expression(elt, {}, path).data for elt in sequence.elts]
    if any(type(value) not in (int, float) for value in literal_angles):
        raise ValueError("The full-test domain list contains a dynamic angle.")
    selected_angle = require_unique([value for value in literal_angles if type(value) in (int, float)
                                     and angle_text(value) == domain],
                                    "The requested domain is absent or ambiguous in the full-test recipe.")
    nested = require_unique([node for node in angles.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name)],
                            "Cannot identify one per-image test loop.")
    raw = dotted(nested.iter)
    loads = []
    for branch in init.body:
        if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare) or len(branch.test.ops) != 1:
            continue
        value = branch.test.comparators[0]
        if not isinstance(branch.test.ops[0], ast.Eq) or not isinstance(value, ast.Constant) or value.value != dataset:
            continue
        for target, expression, site in assigned(branch.body):
            if (isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 4 and dotted(target.elts[2]) == raw
                    and call_is(expression, "load_" + dataset) and not expression.args and not expression.keywords):
                loads.append((branch.test, site))
    condition, load = require_unique(loads, "Cannot link the images to the selected dataset's test split.")
    append = require_unique([call for call in straight_line_calls(nested.body)
                             if call_is(call, buffer + ".append") and len(call.args) == 1],
                            "Cannot identify one append into the evaluation image buffer.")
    transform = append.args[0]
    trail.extend(tracer.location(path, node) for node in (condition, load, sequence, append, stored))
    if (not isinstance(transform, ast.Call) or len(transform.args) != 2
            or dotted(transform.args[0]) != nested.target.id):
        raise ValueError("The evaluation buffer receives original images or an unsupported transformation.")
    actual = tracer.expression(transform.args[1], {angles.target.id: Value(selected_angle)}, path).data
    if type(actual) not in (int, float):
        raise ValueError("The angle passed to the rotation wrapper is dynamic.")
    resets = [site for dest, expr, site in assigned(prior) if site.lineno < angles.lineno
              and isinstance(dest, (ast.Tuple, ast.List)) and isinstance(expr, (ast.Tuple, ast.List))
              and any(dotted(k) == buffer and isinstance(v, ast.List) and not v.elts for k, v in zip(dest.elts, expr.elts))]
    resets.extend(site for dest, expr, site in assigned(prior) if site.lineno < angles.lineno
                  and dotted(dest) == buffer and isinstance(expr, ast.List) and not expr.elts)
    reset = require_unique(resets[-1:], "Cannot establish a clean test-image buffer before rotation.")
    nodes = [node for stmt in scope[scope.index(reset) + 1:scope.index(stored)] for node in ast.walk(stmt)]
    writes = [node for node in nodes if isinstance(node, ast.Call) and dotted(node.func).startswith(buffer + ".")]
    if writes != [append] or any(
        isinstance(node, (ast.Name, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del))
        and (dotted(node.value) if isinstance(node, ast.Subscript) else dotted(node)) == buffer for node in nodes
    ):
        raise ValueError("The full-test array may include a competing image producer.")
    angle, operation, site, returned = rotation_wrapper(tracer, path, transform, actual)
    trail.extend(tracer.location(path, node) for node in (reset, site, returned))
    return {"value": angle, "use": tracer.location(path, operation), "storage": "in_memory",
            "definitions": [tracer.location(path, sequence)], "stage_trace": trail,
            "scope": ("Fresh recipe with no pre-existing local cache." if uses_cache
                      else "Documented full-test evaluation.")}


def session_findings(snapshot, pdf, dataset, stage, domain):
    choices = recipes(snapshot, dataset, stage, study_method(pdf))
    if not choices:
        return [], [], None
    tracer = MemoryTrace(snapshot, domain)
    if len(choices) != 1:
        return [], choices, "Multiple documented in-memory evaluation recipes match the selected experiment."
    recipe = choices[0]
    trail = []
    try:
        path, cls = class_route(tracer, recipe, trail)
        field = batch_feed(tracer, path, cls, trail)
        return [image_buffer(tracer, path, cls, field, domain, recipe["arguments"]["--dataset"], trail)], choices, None
    except (ValueError, SyntaxError) as error:
        recipe["stage_trace"] = trail
        return [], choices, str(error)
