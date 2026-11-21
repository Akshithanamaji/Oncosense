from pathlib import Path

files = [
    Path(r"C:\Python313\Lib\site-packages\streamlit\static\static\js\IFrameUtil.PlC_b34Z.js"),
    Path(r".venv\Lib\site-packages\streamlit\static\static\js\IFrameUtil.PlC_b34Z.js"),
]

for f in files:
    if not f.exists():
        continue
    content = f.read_text(encoding="utf-8")
    deprecated = [
        "`ambient-light-sensor`,",
        "`battery`,",
        "`document-domain`,",
        "`layout-animations`,",
        "`legacy-image-formats`,",
        "`oversized-images`,",
        "`vr `,",
        "`wake-lock`,",
    ]
    for d in deprecated:
        content = content.replace(d, "")
    f.write_text(content, encoding="utf-8")
    print("SUCCESS: Cleaned deprecated permissions from", f)

