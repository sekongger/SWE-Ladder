"""OpenHands funnel v2: adds bash-side script-creation detection (heredoc/redirect/touch/tee)
so reproduce/verify are no longer undercounted for openhands trajectories that author repro
scripts via execute_bash instead of str_replace_editor create. r2egym family numbers unchanged
(their models use editor create; v1 analyzer remains canonical there)."""
import json, re, os
DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
FAIL_RE = re.compile(r"Traceback \(most recent call last\)|AssertionError|FAILED|Error:|error:|Exception|assert(ion)? failed|exit code [1-9]", re.I)
LT = chr(60)
FUNC_RE = re.compile(LT + r"function=(\w+)" + chr(62))
PP = re.compile(LT + r"parameter=path" + chr(62) + r"\s*([^<\s]+)\s*" + LT + r"/parameter")
PC = re.compile(LT + r"parameter=command" + chr(62) + r"\s*([\w-]+)")
BASH_CREATE_RE = re.compile(r"(?:^|[;&|]\s*)(?:cat|printf|echo)\s+.*?\>\s*'?\"?([^\s'\"<>|&;]+\.py)", re.S)
TOUCH_RE = re.compile(r"(?:^|[;&|]\s*)touch\s+'?\"?([^\s'\"<>|&;]+\.py)")
TEE_RE = re.compile(r"\btee\s+'?\"?([^\s'\"<>|&;]+\.py)")
EDITORS = ("file_editor", "str_replace_editor")
def norm(p):
    p = p.strip()
    return p[len("/testbed/"):] if p.startswith("/testbed/") else p.lstrip("./")
def is_test_path(p):
    p = p.lower()
    return p.startswith("test") or "/test" in p or p.endswith("_test.py") or p.endswith("_tests.py")
def diff_files(t):
    return {norm(m.group(2)) for m in DIFF_RE.finditer(t or "")}
def iter_records(path):
    DEC = json.JSONDecoder()
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1
def analyze(rec):
    ds = rec.get("ds") or {}
    if isinstance(ds, str):
        try: ds = json.loads(ds)
        except Exception: ds = {}
    gold = {f for f in diff_files(ds.get("patch", "")) if not is_test_path(f)}
    edited = diff_files(rec.get("output_patch", ""))
    viewed, created = set(), set()
    ev = []; last_edit = -1
    for i, s in enumerate(rec.get("trajectory_steps") or []):
        act = s.get("action") or ""; obs = s.get("observation") or ""
        m = FUNC_RE.search(act); fn = m.group(1) if m else ""
        pm = PP.search(act); path = norm(pm.group(1)) if pm else None
        cm = PC.search(act); sub = cm.group(1) if cm else ""
        if fn in EDITORS + ("search", "grep") and path: viewed.add(path)
        if (fn == "create" or (fn in EDITORS and sub == "create")) and path: created.add(path)
        if fn == "execute_bash":
            cmd = act
            for rx in (BASH_CREATE_RE, TOUCH_RE, TEE_RE):
                for g in rx.findall(cmd): created.add(norm(g))
        if fn == "str_replace" or (fn in EDITORS and sub in ("str_replace", "insert")):
            if path and path not in created and not is_test_path(path): last_edit = i
        if fn == "execute_bash":
            rc = any(os.path.basename(c) in act or c in act for c in created)
            ev.append((i, rc, bool(FAIL_RE.search(obs))))
    reproduce = any(rc and fl for _, rc, fl in ev)
    verify = any(i > last_edit and rc and not fl for i, rc, fl in ev) if last_edit >= 0 else False
    return dict(localize=bool(gold & (viewed | edited)),
                loc_recall=(len(gold & edited) / len(gold)) if gold else None,
                reproduce=reproduce, patch_nonempty=bool((rec.get("output_patch") or "").strip()),
                verify=verify, solve=bool(rec.get("reward")), n_steps=len(rec.get("trajectory_steps") or []))
