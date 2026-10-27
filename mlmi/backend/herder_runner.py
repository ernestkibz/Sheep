"""Runs one herder in its own process: document in, document out, nothing else.

    python -I herder_runner.py  < {"code", "doc", "args", "values"}  > {"doc", "result", "log"} | {"error"}

A herder is code a model wrote. It gets the document as plain data and a small
set of builtins; it has no open(), no imports beyond pure-computation modules,
no attribute access to anything starting with "_", and its process is killed
when its time is up. What it returns is only ever saved through the
workspace, so the worst a herder can do is a bad edit -- and that has an undo.
"""
import ast
import builtins
import json
import sys

ALLOWED_MODULES = {"math", "re", "json", "statistics", "datetime", "random", "itertools", "collections",
                   "string", "textwrap", "decimal", "fractions", "functools", "operator", "copy"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "hasattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
                "super", "type", "object", "dir", "classmethod", "staticmethod", "property", "__builtins__",
                "__loader__", "__spec__", "sys", "os", "subprocess", "builtins", "importlib"}
SAFE = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "next", "print", "range", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip", "divmod", "pow", "chr", "ord", "repr", "iter",
        "ValueError", "KeyError", "IndexError", "TypeError", "Exception", "ZeroDivisionError", "StopIteration",
        "ArithmeticError", "LookupError", "RuntimeError"]


def check(code):
    """Refuse code that could reach past the document. Returns a list of reasons (empty = fine)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error on line {e.lineno}: {e.msg}"]
    why = []
    has_herd = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "herd":
            has_herd = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_MODULES:
                    why.append(f"line {node.lineno}: import of '{m}' is not allowed "
                               f"(allowed: {', '.join(sorted(ALLOWED_MODULES))})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            why.append(f"line {node.lineno}: '.{node.attr}' is not allowed (no names starting with _)")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            why.append(f"line {node.lineno}: '{node.id}' is not allowed in a herder")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            why.append(f"line {node.lineno}: global/nonlocal are not allowed")
    if not has_herd:
        why.append("a herder must define: def herd(doc, args):")
    return why


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in a herder")
    return __import__(name, globals, locals, fromlist, level)


def main():
    req = json.loads(sys.stdin.read())
    code = req["code"]
    bad = check(code)
    if bad:
        print(json.dumps({"error": "; ".join(bad)}))
        return
    log = []
    safe = {k: getattr(builtins, k) for k in SAFE}
    safe["print"] = lambda *a, **k: log.append(" ".join(str(x) for x in a))
    safe["__import__"] = _import
    safe["__build_class__"] = builtins.__build_class__
    helpers = req.get("helpers") or {}
    g = {"__builtins__": safe, "__name__": "herder", "VALUES": req.get("values") or {}}
    exec(compile(helpers.get("prelude", ""), "<helpers>", "exec"), g)
    exec(compile(code, "<herder>", "exec"), g)
    doc = req["doc"]
    try:
        res = g["herd"](doc, req.get("args") or {})
    except Exception as e:  # the model's code failed: tell it where
        import traceback
        tb = traceback.extract_tb(sys.exc_info()[2])
        line = next((f.lineno for f in reversed(tb) if f.filename == "<herder>"), "?")
        print(json.dumps({"error": f"{type(e).__name__} on line {line}: {e}", "log": log[-20:]}))
        return
    if isinstance(res, dict) and res.get("sheep") in ("words", "sheets", "presentations"):
        doc, res = res, None
    try:
        out = json.dumps({"doc": doc, "result": res, "log": log[-50:]}, ensure_ascii=False)
    except (TypeError, ValueError):
        out = json.dumps({"doc": doc, "result": str(res), "log": log[-50:]}, ensure_ascii=False)
    print(out)


if __name__ == "__main__":
    main()
