"""Replace deprecated st.components.v1.html with st.iframe in oncosense_common.py."""
from pathlib import Path

p = Path("oncosense_common.py")
src = p.read_text(encoding="utf-8")

# Remove the now-unused import alias for _st_components
src = src.replace(
    "import streamlit.components.v1 as _st_components\n",
    "",
)

# Fix the two cookie helper calls that used _st_components.html(...)
src = src.replace(
    '_st_components.html(f"<script>{js}</script>", height=0)',
    'st.iframe(f"<script>{js}</script>", height=0)',
)

# Fix render_intro_loader and render_scroll_effects that used st.components.v1.html
src = src.replace(
    "    st.components.v1.html(",
    "    st.iframe(",
)

p.write_text(src, encoding="utf-8")
print("Done — all st.components.v1.html replaced with st.iframe")
