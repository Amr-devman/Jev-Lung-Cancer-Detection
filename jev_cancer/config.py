"""Paths, model settings and the neutral vocabulary.

The model receives pixels and a fixed clinical question bank. It never receives
a diagnosis name, a folder name, a filename, or any metadata. Class folders are
read only to build the stratified evaluation subset and, after predictions are
frozen, to grade them.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

MODEL = "jev-1.14.0"
MAX_SIDE = 1568
DANGER_ZONE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
USD_PER_BTOK = 42.0

# The default subset fraction nets 3,000 images out of the 15,000-image
# dataset: 1,000 per class. ``scripts/01_run.py --fraction`` overrides it and
# ``--seed`` makes the random draw reproducible.
DEFAULT_FRACTION = 0.2
DEFAULT_SEED = 20261020
DEFAULT_WORKERS = 12

# Internal, neutral identifiers. The profile-to-class mapping is used only by
# the grader; no prompt ever contains a class name.
CLASS_IDS = ("c0", "c1", "c2")
PROFILE_IDS = ("p0", "p1", "p2")
PROFILE_TO_CLASS = {"p0": "c0", "p1": "c1", "p2": "c2"}

SOURCE_CLASSES = ("adenocarcinoma", "benign", "squamous_cell_carcinoma")

# Terms that must never reach a prompt. They either name a diagnosis or encode
# dataset provenance.
BANNED_PROMPT_TERMS = (
    "adenocarcinoma",
    "adeno",
    "squamous",
    "carcinoma",
    "cancer",
    "malignant",
    "benign",
    "tumor",
    "tumour",
    "neoplasm",
    "metastasis",
    "scc",
    "luad",
    "lusc",
    "lc25000",
    "kaggle",
    "rm1000",
    "histopathological",
)
