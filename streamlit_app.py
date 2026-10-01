"""Canonical Streamlit entry point for the complete patient Digital Twin app."""

# app.py contains the established three-file PSG workflow, treatment refinement,
# clinical supplements, follow-up PSG, wearable-device data, and continual learning.
#
# Streamlit reruns this entry point after every widget interaction. A normal
# import only executes app.py once because Python caches imported modules,
# which caused the page to become blank immediately after uploading a file.
from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("app.py")),
    run_name="__main__",
)
