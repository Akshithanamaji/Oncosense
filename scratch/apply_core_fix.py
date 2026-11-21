from pathlib import Path

# --- 1. Patch rC in C:\Python313 streamlit JS ---
js_path = Path(r"C:\Python313\Lib\site-packages\streamlit\static\static\js\index.CcFifQPt.js")
if js_path.exists():
    js_content = js_path.read_text(encoding="utf-8")
    old_rc = "function rC(e){return e===`Escape`||e===`Esc`?`esc`:e.toLowerCase()}"
    new_rc = "function rC(e){return!e?``:e===`Escape`||e===`Esc`?`esc`:e.toLowerCase()}"
    if old_rc in js_content:
        js_content = js_content.replace(old_rc, new_rc, 1)
        js_path.write_text(js_content, encoding="utf-8")
        print("SUCCESS: Patched rC in index.CcFifQPt.js")
    elif new_rc in js_content:
        print("ALREADY PATCHED: rC in index.CcFifQPt.js")
    else:
        print("WARNING: old_rc not found in index.CcFifQPt.js")

# --- 2. Patch starlette_path_security_middleware.py in Python313 and .venv ---
middleware_paths = [
    Path(r"C:\Python313\Lib\site-packages\streamlit\web\server\starlette\starlette_path_security_middleware.py"),
    Path(r".venv\Lib\site-packages\streamlit\web\server\starlette\starlette_path_security_middleware.py"),
]

for mw_path in middleware_paths:
    if not mw_path.exists():
        continue
    mw_content = mw_path.read_text(encoding="utf-8")
    target = '        # Only validate HTTP requests (skip WebSocket, lifespan)\n        if scope["type"] != "http":'
    rewrite_code = (
        '        # Route subpage core requests like /Screening/_stcore/host-config to /_stcore/host-config\n'
        '        path = scope.get("path", "")\n'
        '        if "/_stcore/" in path and not path.startswith("/_stcore/"):\n'
        '            scope["path"] = path[path.find("/_stcore/"):]\n'
        '\n'
        '        # Only validate HTTP requests (skip WebSocket, lifespan)\n'
        '        if scope["type"] != "http":'
    )
    if 'if "/_stcore/" in path' in mw_content:
        print(f"ALREADY PATCHED: {mw_path}")
    elif target in mw_content:
        mw_content = mw_content.replace(target, rewrite_code, 1)
        mw_path.write_text(mw_content, encoding="utf-8")
        print(f"SUCCESS: Patched {mw_path}")
    else:
        print(f"WARNING: target not found in {mw_path}")

