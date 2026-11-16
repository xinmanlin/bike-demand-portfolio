"""Execute the walkthrough notebook and export a self-contained HTML copy.

    python report/execute_notebook.py

The notebook only reads files from results/, so this is fast and deterministic.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
NB = NBDIR / "01_results_walkthrough.ipynb"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=180, kernel_name="python3",
                         resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, NB)

subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
     "--embed-images", str(NB), "--output-dir", str(NBDIR)],
    check=True,
)
n_outputs = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
print(f"Executed {NB.name}: {len(nb.cells)} cells, {n_outputs} outputs; HTML exported.")
