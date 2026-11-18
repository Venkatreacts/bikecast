"""Core-code excerpts of the notebooks for Appendix A: keeps model and algorithm code, drops plotting,
printing and display statements (comments are kept). The full notebooks are linked from the appendix."""
import ast
import json
import re
from pathlib import Path

NOISE_CALLS = {"print", "display", "savefig", "show"}
NOISE_ROOTS = {"plt", "ax", "axes", "fig", "sns", "axs", "ax1", "ax2", "ax3", "g"}


def _is_noise(stmt) -> bool:
    """Statements that only draw, print or display."""
    node = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
    if isinstance(stmt, ast.Assign) and isinstance(node, ast.Call):
        f = node.func
        root = f
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in {"plt", "sns"}  # fig, ax = plt.subplots(...)
    if isinstance(stmt, ast.Expr) and isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id in NOISE_CALLS
        root = f
        while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
            root = root.value if not isinstance(root, ast.Call) else root.func
        name = f.attr if isinstance(f, ast.Attribute) else ""
        return (isinstance(root, ast.Name) and root.id in NOISE_ROOTS) or name in NOISE_CALLS
    return False


def strip_noise(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    drop, kept = set(), set()

    def visit(body, force_keep=False):
        noise = [st for st in body if _is_noise(st)] if not force_keep else []
        if len(noise) == len(body):  # block would become empty: keep it whole
            noise = []
        for st in body:
            if st in noise:
                drop.update(range(st.lineno, st.end_lineno + 1))
                continue
            subs = [getattr(st, f, None) for f in ("body", "orelse", "finalbody")]
            subs = [s for s in subs if isinstance(s, list) and s and isinstance(s[0], ast.stmt)]
            first_sub = min((s[0].lineno for s in subs), default=st.end_lineno + 1)
            kept.update(range(st.lineno, first_sub))  # the statement's own header lines
            for s in subs:
                visit(s)
            if not subs:
                kept.update(range(st.lineno, st.end_lineno + 1))

    visit(tree.body)
    drop -= kept  # a line shared with real code (a; b on one line) stays
    lines = [ln for i, ln in enumerate(src.splitlines(), 1) if i not in drop]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def cells(nb_path: Path, wanted: list[str]):
    """(heading, cleaned code) for the first code cell under markdown headings whose number is in `wanted`."""
    nb = json.loads(nb_path.read_text(encoding="utf8"))
    head = None
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("#"):
            head = src.strip().splitlines()[0].lstrip("# ").replace("*", "").replace("`", "").strip()
            shown = False
        elif c["cell_type"] == "code" and head and head.split(".")[0] in wanted:
            code = strip_noise(src)
            if code and not shown:  # first code cell of a section holds the algorithm; later ones are checks/plots
                yield head, code
                shown = True
