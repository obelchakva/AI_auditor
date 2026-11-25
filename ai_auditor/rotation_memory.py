"""Static provenance for a saved experiment configuration and in-memory test images."""

import ast
import re
from pathlib import PurePosixPath

from .initial_lr import compact, main_body, open_article, straight_line_calls
from .rotation import UNKNOWN, CallableSource, StaticTrace, Value, angle_text, dotted, key

DOMAIN_LABEL = r"[A-Za-z]*([+-]?\d+(?:\.\d+)?)\s*[°◦]"
EXTRA = r"(?:unlabelled|unlabeled|unsupervised)"
REPLICATE = "<replicate>"


def variant_paper(pdf, dataset, domain, extra, table):
    definitions, variants, partial = [], [], []
    with open_article(pdf) as article:
        for page in article:
            blocks = [compact(b[4]) for b in page.get_text("blocks") if b[6] == 0]
            for block in blocks:
                for match in re.finditer(rf"\btest(?:ed)?\s+on\s+{DOMAIN_LABEL}", block, re.I):
                    if key(dataset) in key(block) and angle_text(match[1]) == domain:
                        definitions.append({"page": page.number + 1, "excerpt": block, "value": angle_text(match[1])})
            for i, block in enumerate(blocks):
                if not re.match(rf"^Table\s+{table}[.:]", block, re.I):
                    continue
                neighbors = blocks[i + 1:i + 4]
                # Captions can span blocks. Stop at the first results block.
                continuation = []
                for item in neighbors:
                    if re.match(r"Table\s+\d+[.:]|(?:Unsupervised|Unlabeled) domain", item, re.I):
                        break
                    continuation.append(item)
                caption = " ".join([block, *continuation])
                if not re.search(EXTRA, caption, re.I):
                    continue
                citation = {"page": page.number + 1, "excerpt": caption, "table": table}
                partial.append(citation)
                row = next((item for item in neighbors if re.match(r"(?:Unsupervised|Unlabeled) domain", item, re.I)), "")
                if not any(angle_text(x) == extra for x in re.findall(DOMAIN_LABEL, row)):
                    continue
                for context in blocks[max(0, i - 4):i]:
                    test = re.search(rf"\b{DOMAIN_LABEL}\s*is the test domain\b", context, re.I)
                    unlabeled = re.findall(rf"{EXTRA}\s+data\s+from\s+{DOMAIN_LABEL}", context, re.I)
                    if test and angle_text(test[1]) == domain and any(angle_text(x) == extra for x in unlabeled):
                        if (any(angle_text(x) == extra for x in re.findall(DOMAIN_LABEL, caption))
                                and any(angle_text(x) == domain for x in re.findall(DOMAIN_LABEL, caption))):
                            variants.append({**citation, "variant_excerpt": context, "row_excerpt": row,
                                             "additional_unlabeled_domain": extra, "test_domain": domain})
    if len(definitions) != 1 or len(variants) != 1 or variants[0]["page"] - definitions[0]["page"] not in (0, 1):
        return [], definitions + partial + variants
    return [{**definitions[0], "variant_evidence": variants[0]}], []


class MemoryTrace(StaticTrace):
    def expression(self, node, env, path, context=(), uncertain=False):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None):
            value = self.expression(node.left, env, path).data
            if value is not UNKNOWN:
                return Value((value is None) == isinstance(node.ops[0], ast.Is))
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, env, path)
            index = self.expression(node.slice, env, path).data
            if isinstance(base.data, tuple) and type(index) is int and -len(base.data) <= index < len(base.data):
                return Value(base.data[index], base.trace)
        return super().expression(node, env, path, context, uncertain)


def require_unique(items, message):
    if len(items) != 1:
        raise ValueError(message)
    return items[0]


def entry(tracer, path):
    return require_unique([body for node in tracer.document(path)[1].body if (body := main_body(node))],
                          "No unique guarded entry block in the selected script.")


def assigned(body):
    return [(statement.targets[0], statement.value, statement) for statement in body
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1]


def call_is(call, name):
    return isinstance(call, ast.Call) and dotted(call.func) == name


def config_name(tracer, path, expression, body, env):
    # Resolve local filename expressions preceding the read/write. Only literal
    # assignments and simple concatenation/conversions are interpreted.
    for target, value, _ in assigned(body):
        name = dotted(target)
        if name:
            env[name] = tracer.expression(value, env, path)
    result = tracer.expression(expression, env, path).data
    return str(PurePosixPath(result)) if isinstance(result, str) else None


def extra_training(tracer, path, body, extra):
    # Confirm the additional domain is in the unlabelled loader slot fed to
    # a local training function, rather than inferred from a directory name.
    proofs = []
    module = tracer.document(path)[1].body
    entries = {fn.name: fn for fn in module if isinstance(fn, ast.FunctionDef)}
    all_calls = [n for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Call)]
    for target, value, site in assigned(body):
        if not isinstance(target, ast.Subscript) or not isinstance(target.slice, ast.Constant):
            continue
        slot = target.slice.value
        if not isinstance(slot, str) or not re.fullmatch(r"unsup(?:ervised)?|unlabeled|unlabelled", slot, re.I):
            continue
        loader = dotted(value)
        for dest, expr, loader_site in assigned(body):
            if dotted(dest) != loader or not isinstance(expr, ast.Call) or not dotted(expr.func).endswith(".DataLoader"):
                continue
            if not expr.args or not isinstance(expr.args[0], ast.Call):
                continue
            construct = expr.args[0]
            literals = [tracer.expression(x, {}, path).data for x in construct.args]
            if not any(isinstance(x, tuple) and x == (extra,) for x in literals):
                continue
            for call in all_calls:
                fn = entries.get(dotted(call.func))
                if fn is None:
                    continue
                for arg, param in zip(call.args, fn.args.args):
                    if dotted(arg) != dotted(target.value):
                        continue
                    if any(isinstance(n, ast.Subscript) and dotted(n.value) == param.arg
                           and isinstance(n.slice, ast.Constant) and n.slice.value == slot for n in ast.walk(fn)):
                        proofs.append((loader_site, site, call))
    return require_unique(proofs, "Cannot trace the selected additional domain to the unlabelled training loader.")


def config_producer(tracer, path, field, extra, domain, expected, trail):
    body = entry(tracer, path)
    parsed = require_unique([(dotted(t), dotted(v.func).rsplit(".", 1)[0], s)
                             for t, v, s in assigned(body) if isinstance(v, ast.Call)
                             and dotted(v.func).endswith(".parse_args") and not v.args and not v.keywords],
                            "Cannot identify the producer's argparse namespace.")
    namespace, parser, parse_site = parsed
    defaults, sites = {}, {}
    for call in straight_line_calls(body):
        if not call_is(call, parser + ".add_argument"):
            continue
        flags = [arg.value[2:].replace("-", "_") for arg in call.args if isinstance(arg, ast.Constant)
                 and isinstance(arg.value, str) and arg.value.startswith("--")]
        options = {kw.arg: kw.value for kw in call.keywords}
        if len(flags) == 1 and "default" in options and not {"dest", "action"} & options.keys():
            name = namespace + "." + flags[0]
            if name in sites:
                raise ValueError("The producer declares duplicate configuration fields.")
            defaults[name] = tracer.expression(options["default"], {}, path)
            sites[name] = call
    key_name = namespace + "." + field
    if key_name not in sites or sites[key_name].lineno >= parse_site.lineno:
        raise ValueError("The selected test-domain field has no literal argparse default.")
    trail.append(tracer.location(path, sites[key_name]))
    save = require_unique([c for c in straight_line_calls(body) if call_is(c, "torch.save")
                           and len(c.args) == 2 and not c.keywords and dotted(c.args[0]) == namespace],
                          "Cannot identify one save of the producer's configuration namespace.")
    trail.append(tracer.location(path, save))
    prior = [stmt for stmt in body if stmt.lineno < save.lineno]
    # Only the scalar-to-singleton wrapping of the selected field is supported.
    for stmt in prior:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Attribute) and dotted(node) == key_name and isinstance(node.ctx, (ast.Store, ast.Del)):
                wrap = (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and stmt.targets[0] is node
                        and isinstance(stmt.value, (ast.List, ast.Tuple)) and len(stmt.value.elts) == 1
                        and dotted(stmt.value.elts[0]) == key_name)
                if not wrap:
                    raise ValueError("The test-domain configuration field has an unsupported update.")
                trail.append(tracer.location(path, stmt))
    replicates = [name for name, val in defaults.items() if type(val.data) is int] if REPLICATE in expected else [None]
    hits = []
    for name in replicates:
        env = defaults.copy()
        if name:
            env[name] = Value(REPLICATE)
        prefix = save.args[1].left if isinstance(save.args[1], ast.BinOp) else save.args[1]
        filename = config_name(tracer, path, save.args[1], prior, env)
        if filename == expected:
            hits.append((env.get(key_name, Value()).data, name, prefix))
    value, replicate, prefix = require_unique(hits, "The producer and evaluator do not resolve to one matching configuration name.")
    selected = value[0] if isinstance(value, tuple) and len(value) == 1 else value
    if type(selected) not in (str, int) or str(selected) != domain:
        raise ValueError("The saved configuration field does not identify the selected test domain.")
    if replicate:
        trail.append(tracer.location(path, sites[replicate]))
    trail.extend(tracer.location(path, node) for node in extra_training(tracer, path, body, extra))
    trail.extend(tracer.location(path, stmt) for target, _, stmt in assigned(prior) if dotted(target) == dotted(prefix))
    return value


def evaluation_scope(tracer, path):
    body = entry(tracer, path)
    scopes = [(body, {})]
    for stmt in body:
        if (isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name)
                and call_is(stmt.iter, "range") and len(stmt.iter.args) == 1
                and isinstance(stmt.iter.args[0], ast.Constant) and type(stmt.iter.args[0].value) is int
                and stmt.iter.args[0].value > 0 and not stmt.iter.keywords):
            scopes.append((stmt.body, {stmt.target.id: Value(REPLICATE)}))
    return scopes


def evaluation_loader(tracer, path, field, trail):
    options = []
    for body, env in evaluation_scope(tracer, path):
        for target, load, site in assigned(body):
            if not isinstance(target, ast.Name) or not call_is(load, "torch.load") or len(load.args) != 1 or load.keywords:
                continue
            before = [stmt for stmt in body if stmt.lineno < site.lineno]
            filename = config_name(tracer, path, load.args[0], before, env.copy())
            if not filename or not filename.endswith(".config"):
                continue
            config_ref = target.id + "." + field
            later = [stmt for stmt in body if stmt.lineno > site.lineno]
            for dest, expression, loader in assigned(later):
                if not isinstance(dest, ast.Name) or not isinstance(expression, ast.Call) or not dotted(expression.func).endswith(".DataLoader"):
                    continue
                construct = expression.args[0] if expression.args else None
                if not isinstance(construct, ast.Call) or not any(dotted(arg) == config_ref for arg in construct.args):
                    continue
                options.append((filename, site, loader, construct, dest.id, later, [s for _, _, s in assigned(before)]))
    filename, load, loader, construct, name, later, parents = require_unique(
        options, "Cannot trace one selected configuration into an evaluation loader.")
    trail.extend(tracer.location(path, node) for node in [*parents, load, loader])
    if not any(kw.arg == "train" and isinstance(kw.value, ast.Constant) and kw.value.value is False
               for kw in construct.keywords):
        raise ValueError("The evaluation loader is not explicitly constructed with train=False.")
    config_namespace = dotted(load.targets[0])
    ref = config_namespace + "." + field
    for stmt in later:
        if stmt.lineno >= loader.lineno:
            break
        if any((isinstance(n, ast.Attribute) and dotted(n) == ref or isinstance(n, ast.Name) and n.id == config_namespace)
               and isinstance(n.ctx, (ast.Store, ast.Del)) for n in ast.walk(stmt)):
            raise ValueError("The evaluator can overwrite the selected configuration field before loading data.")
    module_env = {}
    tracer.bind_module(path, module_env)
    target = module_env.get(dotted(construct.func), Value())
    if not isinstance(target.data, CallableSource) or not isinstance(target.data.node, ast.ClassDef):
        raise ValueError("Cannot resolve the local dataset class used for evaluation.")
    used = []
    after_loader = [stmt for stmt in later if stmt.lineno > loader.lineno]
    for call in straight_line_calls(after_loader):
        scorer = module_env.get(dotted(call.func), Value()).data
        if not isinstance(scorer, CallableSource) or not isinstance(scorer.node, ast.FunctionDef):
            continue
        for arg, parameter in zip(call.args, scorer.node.args.args):
            if dotted(arg) != name:
                continue
            if any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del))
                   for stmt in after_loader if stmt.lineno < call.lineno for n in ast.walk(stmt)):
                continue
            classifiers = {formal.arg for actual, formal in zip(call.args, scorer.node.args.args)
                           if actual is not arg and dotted(actual)}
            for loop in ast.walk(scorer.node):
                if not isinstance(loop, ast.For) or dotted(loop.iter) != parameter.arg:
                    continue
                pixel = dotted(loop.target.elts[0]) if isinstance(loop.target, (ast.Tuple, ast.List)) else dotted(loop.target)
                consumers = [n for n in straight_line_calls(loop.body) if dotted(n.func) in classifiers
                             and any(dotted(x) == pixel for x in n.args)]
                if consumers:
                    used.append((call, loop.iter, consumers[0], scorer.path))
    call, iteration, consume, scorer_path = require_unique(used, "The evaluation does not consume the selected loader's images.")
    trail.extend([tracer.location(path, call), tracer.location(scorer_path, iteration), tracer.location(scorer_path, consume)])
    return filename, target, construct, ref


def method(cls, name):
    return require_unique([node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name],
                          f"Cannot resolve unique dataset method {name}.")


def test_branch(body, tracer, path, env, trail, guarded):
    for stmt in body:
        if isinstance(stmt, ast.If):
            value = tracer.expression(stmt.test, env, path).data
            if value is UNKNOWN:
                if guarded:
                    raise ValueError("Dataset flow depends on an unresolved branch.")
                yield stmt
                continue
            trail.append(tracer.location(path, stmt.test))
            yield from test_branch(stmt.body if value else stmt.orelse, tracer, path, env, trail, guarded)
        else:
            yield stmt


def first_value(node, index):
    return node.elts[index] if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) > index else None


def constructor_fields(tracer, path, cls, call, ref, config, trail):
    init = method(cls, "__init__")
    params = init.args.args
    if not params or len(call.args) >= len(params):
        raise ValueError("Cannot bind evaluation dataset parameters.")
    params = [p.arg for p in params[1:]]
    env = {name: tracer.expression(value, {}, path) for name, value in
           zip(params[len(params) - len(init.args.defaults):], init.args.defaults)}
    if (init.args.vararg or init.args.kwarg or any(kw.arg not in params or kw.arg in params[:len(call.args)]
                                                 for kw in call.keywords)):
        raise ValueError("The evaluation dataset constructor arguments are unsupported.")
    provided = [(param, value) for param, value in zip(params, call.args)]
    provided.extend((kw.arg, kw.value) for kw in call.keywords)
    for param, node in provided:
        env[param] = Value(config) if dotted(node) == ref else tracer.expression(node, {}, path)
    result = []
    for stmt in test_branch(init.body, tracer, path, env, trail, True):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if (isinstance(stmt.value, ast.Call) and dotted(stmt.value.func).startswith("self.")
                and not stmt.value.args and not stmt.value.keywords):
            result.append((target, stmt.value.func.attr, stmt))
        elif dotted(target).startswith("self."):
            env[dotted(target)] = tracer.expression(stmt.value, env, path)
    target, function, site = require_unique(result, "Cannot trace a unique test-data initializer.")
    trail.append(tracer.location(path, site))
    return target, method(cls, function), env


def stored_pixel(tracer, path, cls, target, env, trail):
    fields = [dotted(x) for x in target.elts] if isinstance(target, (ast.Tuple, ast.List)) else [dotted(target)]
    item = method(cls, "__getitem__")
    index = item.args.args[1].arg if len(item.args.args) == 2 else None
    seen = {}
    for stmt in test_branch(item.body, tracer, path, env, trail, True):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and dotted(stmt.targets[0]):
            seen[dotted(stmt.targets[0])] = (stmt.value, stmt)
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            value = first_value(stmt.value, 0) if isinstance(stmt.value, (ast.Tuple, ast.List)) else stmt.value
            expr, site = seen.get(dotted(value), (value, stmt))
            if isinstance(expr, ast.Subscript) and dotted(expr.slice) == index and dotted(expr.value) in fields:
                trail.extend(tracer.location(path, n) for n in (site, stmt))
                return fields.index(dotted(expr.value))
    raise ValueError("The evaluation dataset does not return images from the selected test-data field.")


def rotate_call(expression, conversions):
    while isinstance(expression, ast.Call) and dotted(expression.func) in conversions:
        if len(expression.args) != 1 or expression.keywords:
            return None
        expression = expression.args[0]
    if isinstance(expression, ast.Call) and dotted(expression.func) in {
        "transforms.functional.rotate", "torchvision.transforms.functional.rotate",
    }:
        return expression
    return None


def input_source(expression, sources, conversions):
    while isinstance(expression, ast.Call) and dotted(expression.func) in conversions:
        if len(expression.args) != 1 or expression.keywords:
            return False
        expression = expression.args[0]
    while isinstance(expression, ast.Subscript):
        expression = expression.value
    return dotted(expression) in sources


def pixel_return(tracer, path, method_node, env, position, dataset, trail, partial):
    rotated, conversions, dataset_bindings, raw_pixels = {}, set(), [], set()
    for stmt in test_branch(method_node.body, tracer, path, env, trail, False):
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                break
            expr = first_value(stmt.value, position) if isinstance(stmt.value, (ast.List, ast.Tuple)) else stmt.value
            if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "unsqueeze"
                    and len(expr.args) == 1 and not expr.keywords):
                expr = expr.func.value
            match = rotated.get(dotted(expr)) or direct_rotation(tracer, path, expr, env, conversions, raw_pixels, partial)
            if match and match["value"] is None:
                raise ValueError("The angle passed to the selected test rotation is dynamic or unresolved.")
            if match and len(dataset_bindings) == 1:
                trail.extend([dataset_bindings[0], tracer.location(path, stmt)])
                return match
            break
        if isinstance(stmt, ast.If) and any(isinstance(node, (ast.Return, ast.Raise)) for node in ast.walk(stmt)):
            raise ValueError("An unresolved dataset branch can bypass the test-image return.")
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            name = dotted(stmt.targets[0])
            tensor = stmt.value
            if name and isinstance(tensor, ast.Call) and dotted(tensor.func) in {
                "transforms.ToTensor", "transforms.ToPILImage",
            } and not tensor.args and not tensor.keywords:
                conversions.add(name)
            dataset_calls = [item for item in ast.walk(tensor) if isinstance(item, ast.Call)
                             and dotted(item.func).startswith("datasets.")
                             and key(dataset).endswith(key(dotted(item.func).split(".")[-1]))]
            from_dataset = bool(dataset_calls) or input_source(tensor, raw_pixels, conversions)
            dataset_bindings.extend(tracer.location(path, item) for item in dataset_calls)
            direct = direct_rotation(tracer, path, tensor, env, conversions, raw_pixels, partial)
            if name:
                if from_dataset:
                    raw_pixels.add(name)
                else:
                    raw_pixels.discard(name)
            if direct:
                direct["stage_trace"].append(tracer.location(path, stmt))
                rotated[name] = direct
            elif dotted(tensor) in rotated:
                rotated[name] = rotated[dotted(tensor)]
            else:
                rotated.pop(name, None)
            env[name] = tracer.expression(tensor, env, path)
            if env[name].data is not UNKNOWN:
                env[name] = Value(env[name].data, (*env[name].trace, tracer.location(path, stmt)))
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and call_is(stmt.iter, "range"):
            # For an indexed buffer, require one unconditional write per
            # source index and an iteration count from that same source.
            if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Assign) and len(stmt.body[0].targets) == 1:
                write = stmt.body[0]
                target = write.targets[0]
                rotation = rotate_call(write.value, conversions)
                if (isinstance(target, ast.Subscript) and dotted(target.slice) == stmt.target.id and rotation
                        and len(stmt.iter.args) == 1 and call_is(stmt.iter.args[0], "len")
                        and len(stmt.iter.args[0].args) == 1 and rotation.args):
                    source = rotation.args[0]
                    if isinstance(source, ast.Call) and dotted(source.func) in conversions and len(source.args) == 1:
                        source = source.args[0]
                    if (isinstance(source, ast.Subscript) and dotted(source.slice) == stmt.target.id
                            and dotted(source.value) and dotted(source.value) == dotted(stmt.iter.args[0].args[0])):
                        item = direct_rotation(tracer, path, write.value, env, conversions, raw_pixels, partial)
                        if item:
                            item["stage_trace"].append(tracer.location(path, write))
                            rotated[dotted(target.value)] = item
                            continue
        # Images can also come from the first member of a batch unpacked
        # from the selected dataset loader.
        if isinstance(stmt, ast.For) and input_source(stmt.iter, raw_pixels, conversions):
            propagate_batch(stmt.target, stmt.body, raw_pixels)
        elif (isinstance(stmt, ast.For) and call_is(stmt.iter, "enumerate") and len(stmt.iter.args) == 1
              and input_source(stmt.iter.args[0], raw_pixels, conversions)):
            propagate_batch(first_value(stmt.target, 1), stmt.body, raw_pixels)
        # Any other subsequent write to an angle or candidate buffer makes it
        # unresolved. Plain label writes and untracked data operations are ignored.
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                name = dotted(node.value) if isinstance(node, ast.Subscript) else dotted(node)
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and node is stmt.targets[0]
                        and not isinstance(node, ast.Subscript)):
                    rotated.pop(name, None)
                    env.pop(name, None)
    raise ValueError("Cannot trace a resolved rotation through the value returned as test images.")


def propagate_batch(target, body, raw_pixels):
    pixel = dotted(first_value(target, 0))
    if not pixel:
        return
    for statement in body:
        if isinstance(statement, ast.Assign) and dotted(statement.value) == pixel:
            raw_pixels.update(dotted(item) for item in statement.targets if dotted(item))


def direct_rotation(tracer, path, expr, env, conversions, raw_pixels, partial):
    call = rotate_call(expr, conversions)
    if not call:
        return None
    angles = [call.args[1]] if len(call.args) == 2 else [kw.value for kw in call.keywords if kw.arg == "angle"]
    angle = tracer.expression(angles[0], env, path) if len(angles) == 1 else Value()
    item = {"value": angle_text(angle.data) if type(angle.data) in (int, float) else None,
            "use": tracer.location(path, call), "definitions": list(angle.trace), "stage_trace": []}
    partial.append({**item})
    pixels = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "img"), None)
    return item if input_source(pixels, raw_pixels, conversions) else None


def memory_findings(snapshot, dataset, domain, extra, producer, evaluator, field):
    tracer = MemoryTrace(snapshot, domain)
    config_trace, evaluation_trace, dataset_trace, partial = [], [], [], []
    link = {"source": "user_selected_configuration_recipe", "producer": producer, "evaluator": evaluator,
            "config_field": field, "additional_unlabeled_domain": extra}
    try:
        if producer not in snapshot.files or evaluator not in snapshot.files:
            raise ValueError("The selected producer or evaluator is absent from the pinned commit.")
        filename, target, call, ref = evaluation_loader(tracer, evaluator, field, evaluation_trace)
        link["configuration_pattern"] = filename
        config = config_producer(tracer, producer, field, extra, domain, filename, config_trace)
        path, cls = target.data.path, target.data.node
        target_field, data_method, env = constructor_fields(tracer, path, cls, call, ref, config, dataset_trace)
        position = stored_pixel(tracer, path, cls, target_field, env, dataset_trace)
        result = pixel_return(tracer, path, data_method, env, position, dataset, dataset_trace, partial)
        result["stage_trace"] = [*config_trace, *evaluation_trace, *target.trace, *result["stage_trace"], *dataset_trace]
        result["storage"] = "in_memory"
        return [result], [link], None
    except (ValueError, SyntaxError) as error:
        link["stage_trace"] = [*config_trace, *evaluation_trace, *dataset_trace]
        return partial, [link], str(error)
