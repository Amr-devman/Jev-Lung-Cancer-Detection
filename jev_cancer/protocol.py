"""The clinical question bank.

One request, one image, no references, no labels. Every question is a yes/no or
multiple-choice observation about morphology, and every question is followed by
a plain-language description of what the observation means and what to look for
in the pixels. No prompt names a disease, a dataset folder, or a filename.

Three neutral profiles anchor the classification:

  p0  glandular / mucinous epithelial architecture
  p1  open alveolar architecture
  p2  solid cohesive epithelial architecture with dense cytoplasm

The mapping from profile to diagnosis is in ``config.PROFILE_TO_CLASS`` and is
used only by the grader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .config import BANNED_PROMPT_TERMS, PROFILE_IDS

PROTOCOL_VERSION = "2026-10-21.described"

# Outcome descriptions shared by every yes/no observation. They fix the
# polarity ("yes" = the described property is clearly present) and tell the
# model when to answer no instead of guessing.
NOUL_CRITERIA = {
    "true": "The described property is clearly visible in the image.",
    "false": "The property is absent, too faint to be sure, or only a doubtful hint.",
}


@dataclass(frozen=True)
class Feature:
    """One observable property and the evidence it carries for p0/p1/p2.

    ``ask`` is the yes/no question. ``look`` is a plain-language description of
    what the observation actually means and what to look for in the pixels.
    Weights are a transparent clinical prior, not fitted coefficients.
    """

    id: str
    ask: str
    look: str
    weights: tuple[float, float, float]


FEATURES: tuple[Feature, ...] = (
    # --- Adequacy: gates, not diagnostic evidence ---------------------------
    Feature("q_tissue", "Does `sample` contain intact tissue with visible cells and extracellular structure?",
        "Look for real tissue: groups of cells with visible dark nuclei, plus the pink material that "
        "holds them together. Blank glass, pure blur, or isolated specks do not count. Answer yes "
        "only when there is enough intact tissue to judge.", (0.0, 0.0, 0.0)),
    Feature("q_artifact", "Is more than half of `sample` covered by blood, crush artefact, or structureless debris?",
        "Estimate how much of the picture is unreadable: smears of red blood, tissue crushed into "
        "streaks, or shapeless debris without recognisable cells. Answer yes when these cover more "
        "than half of the picture.", (0.0, 0.0, 0.0)),
    Feature("q_stain", "Are nuclei and cytoplasm distinguishable across most of `sample`?",
        "Look at the colours: nuclei should read as dark blue-purple dots and the cell body as pink. "
        "Answer yes when nucleus and cell body are clearly separable over most of the picture; answer "
        "no when everything is too pale, too dark, or washed out.", (0.0, 0.0, 0.0)),
    # --- Architecture ------------------------------------------------------
    Feature("a_open_spaces", "Does `sample` contain multiple open, air-filled spaces bounded by thin tissue walls?",
        "Look for several empty-looking round or irregular holes in the tissue, like the air sacs of "
        "healthy lung, each surrounded by a thin wall of tissue. The holes look mostly white and empty, "
        "not packed with cells or fluid.", (0.0, 3.0, -1.5)),
    Feature("a_thin_walls", "Are the walls between those open spaces thin, roughly one or two cells thick?",
        "Look at the walls that separate the open spaces. In healthy air sacs the wall is a very fine "
        "line of pink tissue, about one or two cells across, with small dark nuclei spaced along it.",
        (0.0, 2.5, -1.0)),
    Feature("a_architecture_uniform", "Is the tissue architecture orderly and similar across most of `sample`?",
        "Judge the overall layout: is the tissue arranged in a repeating, calm, orderly way across "
        "most of the picture, with similar structures next to each other? Answer yes for a regular "
        "repeating layout, no for a chaotic mixed one.", (-0.5, 2.0, -1.0)),
    Feature("a_glands", "Does `sample` contain rounded or irregular gland-like structures with a central open lumen?",
        "Look for rings of cells around a central open hole (a lumen) that is empty or holds pale "
        "material. Depending on how the tissue was cut the rings appear as circles, ovals, crooked "
        "rings, crescents or short tubes.", (2.5, -1.5, -0.5)),
    Feature("a_papillae", "Does `sample` contain finger-like projections of tissue covered by a layer of cells?",
        "Look for finger-like or tree-like projections of tissue sticking out into an open space: a "
        "thin pink core covered by a tidy layer of cells, like a glove over a finger.", (2.0, -1.0, -0.5)),
    Feature("a_mucin", "Does `sample` contain pools of pale, homogeneous, slightly blue-grey material between cells?",
        "Look for lakes of pale, smooth, slightly blue-grey slimy material between or inside groups of "
        "cells. The pool looks uniform and glassy, with no visible cell structure inside it.",
        (2.5, -0.5, -1.0)),
    Feature("a_solid", "Does `sample` contain broad solid islands or sheets of cells with no open lumen inside them?",
        "Look for large solid sheets or islands made of many cells packed together, with no open hole "
        "in the middle. The island reads as a solid mass of cells, not as a ring around a lumen.",
        (0.5, -1.5, 2.0)),
    Feature("a_desmoplasia", "Is there dense pink fibrous tissue separating groups of cells?",
        "Look for dense pink scar-like tissue sitting between groups of cells. It looks fibrous and "
        "stringy, brighter pink than the cell bodies, and pushes the cell groups apart.", (1.0, -1.5, 1.0)),
    Feature("a_fibrous_bands", "Are there broad bands of dense pink collagen dividing the tissue into compartments?",
        "Look for wide ribbons or bands of dense pink collagen running through the tissue and cutting "
        "it into separate compartments.", (0.8, -1.0, 0.8)),
    Feature("a_single_file", "Do cells run in single-file lines through fibrous tissue?",
        "Look for cells marching one behind the other in single-file lines through fibrous tissue, "
        "like beads on a string.", (1.5, -0.5, -0.5)),
    Feature("a_necrosis", "Do the centres of cell islands contain structureless pink debris or fragmented nuclei?",
        "Look inside large groups of cells at their centre: dead tissue shows as shapeless pink debris "
        "with no intact cells, or as scattered broken dark nuclear fragments.", (0.3, -0.5, 1.5)),
    Feature("a_surface_columnar", "Is there a surface lining of tall cells whose nuclei sit at different heights?",
        "Look for a surface covered by tall narrow cells standing side by side like a picket fence, "
        "with their dark nuclei sitting at different heights so the line of nuclei looks staggered.",
        (1.0, 0.5, -0.5)),
    Feature("a_whorl", "Does `sample` contain a whorl of concentric, brightly pink material inside a group of cells?",
        "Look for a swirling, onion-skin structure inside a group of cells: concentric layers of "
        "bright pink material wrapped around a centre, like the rings of an onion.", (-1.0, -1.0, 2.5)),
    Feature("a_spines", "Are fine spine-like connections visible between neighbouring cells?",
        "Look closely at the narrow gaps between neighbouring cells in a solid group: fine, tiny "
        "spine-like lines bridge the cells across the gaps.", (-0.5, -1.0, 2.0)),
    Feature("a_lepidic", "Do cells line existing air spaces in a single layer without filling the space?",
        "Look for cells growing as a single neat layer along the existing walls of air spaces, like "
        "wallpaper lining a room, without piling up or filling the space.", (1.5, 0.5, -1.0)),
    Feature("a_micropapillary", "Are there small tufts of cells floating in open spaces without a fibrous core?",
        "Look for small tufts or clumps of cells floating freely inside open spaces, with no supporting "
        "tissue core inside the tuft.", (1.5, -0.5, 0.0)),
    Feature("a_cribriform", "Are there sieve-like spaces within a solid group of cells?",
        "Look for a solid group of cells riddled with many small round holes, like a sieve or swiss "
        "cheese, punched through the cell mass.", (1.2, -0.5, 0.0)),
    Feature("a_airspace_filling", "Are open spaces filled by sheets of cells rather than by air or macrophages?",
        "Look at the open spaces: are they packed with sheets of cells, instead of being empty (air) "
        "or holding loose pale scavenger cells?", (1.0, -1.0, 0.5)),
    # --- Cytology ----------------------------------------------------------
    Feature("c_dense_cytoplasm", "Do most cells have abundant, dense, opaque pink cytoplasm?",
        "Look at the cell bodies: most cells have a lot of cytoplasm that is dense and opaque pink, so "
        "the cells look plump and solid rather than mostly nucleus.", (-0.5, -0.5, 2.0)),
    Feature("c_sharp_borders", "Do neighbouring cells have sharply defined borders rather than blending together?",
        "Look at the edges between neighbouring cells: the boundaries are clearly drawn and each cell "
        "reads as a separate tile, instead of the cells melting into one pink mass.", (0.0, -0.5, 2.0)),
    Feature("c_columnar", "Are most cells taller than they are wide?",
        "Look at the shape of most cells: they are taller than wide, like columns, with the nucleus "
        "sitting toward the base of the cell.", (2.0, -1.0, -1.0)),
    Feature("c_vacuoles", "Do cells contain clear rounded vacuoles that push the nucleus to one side?",
        "Look inside the cells for clear round bubbles (vacuoles) in the cell body that push the dark "
        "nucleus off to one side.", (1.5, 0.0, -0.5)),
    Feature("c_signet", "Do cells contain a large pale vacuole that pushes the nucleus to the cell rim?",
        "Look for cells with one large pale bubble filling nearly the whole cell and squashing the "
        "nucleus flat against the cell rim, like a signet ring.", (1.2, 0.0, -0.5)),
    Feature("c_nucleoli", "Do many cells have enlarged, prominent nucleoli?",
        "Look inside the dark nuclei for a clearly visible darker spot in the middle (the nucleolus). "
        "Answer yes when many nuclei contain a large obvious spot.", (1.0, 0.0, 0.5)),
    Feature("c_pleomorphism", "Do nuclei vary markedly in size and shape from one cell to the next?",
        "Compare the nuclei with each other: do they vary a lot in size and shape from cell to cell, "
        "some large and lumpy, some small and round?", (0.5, 0.0, 0.5)),
    Feature("c_hyperchromasia", "Are many nuclei intensely dark?",
        "Look at how dark the nuclei are: many of them are intensely dark, almost black-purple, like "
        "ink dots.", (0.4, 0.0, 0.4)),
    Feature("c_high_nc", "Do many cells have a nucleus that fills most of the cell?",
        "Look at the balance of nucleus to cell body: in many cells the dark nucleus fills nearly the "
        "whole cell, leaving only a thin rim of pink.", (0.3, 0.0, 0.3)),
    Feature("c_mitoses", "Are mitotic figures visible?",
        "Look for cells caught in division (mitotic figures): dense clumps of dark chromatin that look "
        "like starbursts, fuzzy spiders, or split dark blobs instead of smooth round nuclei.",
        (0.3, 0.0, 0.3)),
    Feature("c_nuclear_contour", "Do many nuclei have irregular, angular or folded contours?",
        "Look at the outline of the nuclei: many are not smooth circles but irregular, angular, "
        "notched, or folded in on themselves.", (0.5, 0.0, 0.5)),
    Feature("c_chromatin", "Is the chromatin coarse and clumped in many nuclei?",
        "Look at the texture inside the nuclei: it reads as coarse and clumped, like grains of dark "
        "sand, instead of fine and even.", (0.4, 0.0, 0.4)),
    Feature("c_hobnail", "Do some cells bulge into an open space with the nucleus at the outer tip?",
        "Look at the cells lining an open space: some bulge into it like a thumb pushing into a "
        "balloon, with the nucleus sitting at the bulging outer tip.", (1.5, 0.0, -0.5)),
    Feature("c_polygonal", "Are most cells flattened or polygonal with angular outlines?",
        "Look at the shape of most cells: they are angular, polygonal or flattened, with straight-ish "
        "edges and clear corners, like floor tiles.", (-0.5, 0.0, 1.5)),
    Feature("c_shrunken", "Are there individual cells with shrunken, intensely pink cytoplasm and a small dark nucleus?",
        "Look for single cells that have died and shrunk: small cells with intensely pink dense "
        "cytoplasm and a tiny condensed dark nucleus.", (-0.5, -0.5, 2.0)),
    Feature("c_cilia", "Are there hair-like cilia on the surface of the cells?",
        "Look at the surface of cells lining an open space: fine hair-like lines (cilia) stick out "
        "from the surface like a tiny brush border.", (-0.5, 1.5, -0.5)),
    # --- Background --------------------------------------------------------
    Feature("b_pigment", "Is there black or dark brown granular pigment in the tissue?",
        "Look for dark black or brown granular specks scattered through the tissue or sitting inside "
        "cells, like soot.", (0.0, 1.5, 0.0)),
    Feature("b_rbc", "Are red blood cells present outside blood vessels or within open spaces?",
        "Look for tiny uniform red-orange discs (red blood cells) lying outside blood vessels or loose "
        "inside open spaces.", (0.0, 0.8, 0.0)),
    Feature("b_macrophages", "Are there large cells with pale foamy cytoplasm sitting inside open spaces?",
        "Look inside open spaces for large rounded cells with pale bubbly cytoplasm and a small dark "
        "nucleus: scavenger cells that clear the air spaces.", (0.0, 1.2, 0.0)),
    Feature("b_inflammation", "Is there a dense crowd of small round inflammatory cells?",
        "Look for a dense crowd of small round cells with almost no cytoplasm and dark round nuclei, "
        "packed between the structures.", (0.0, 0.2, 0.0)),
    Feature("b_hemosiderin", "Is there golden-brown granular pigment within large cells?",
        "Look for coarse golden-brown granules sitting inside large cells, left behind by old "
        "bleeding.", (0.0, 0.8, 0.0)),
)

FEATURE_IDS = tuple(feature.id for feature in FEATURES)

PROFILE_TEXT: dict[str, str] = {
    "p0": (
        "Glandular growth. Rounded or irregular rings of cells around an open central hole (a lumen), "
        "lined by tall cells; pale blue-grey mucin pools between or inside the structures; dense pink "
        "scar-like tissue separates the structures. Look for the ring-around-a-hole pattern."
    ),
    "p1": (
        "Air-sac architecture. Several open, mostly empty spaces separated by very thin walls one or "
        "two cells thick; little crowded cell growth; scattered dark pigment specks, red blood cells "
        "and pale bubbly scavenger cells inside the spaces. Look for the open airy pattern."
    ),
    "p2": (
        "Solid cohesive growth. Broad islands or sheets of polygonal cells with abundant dense opaque "
        "pink cytoplasm and sharply drawn cell borders; concentric onion-skin whorls of bright pink "
        "material; dead pink debris in the centre of the islands. Look for the solid sheet pattern."
    ),
}

CELL_SHAPE_TEXT = {
    "columnar_or_cuboidal": "Most cells are taller than wide, or roughly square, with the nucleus toward the base.",
    "polygonal_squamoid": "Most cells are polygonal or flattened with angular outlines and clear corners.",
    "flattened_lining": "Most cells are flat and closely applied to thin tissue walls.",
    "round_dyscohesive": "Most cells are rounded and loosely arranged, without a clear architectural pattern.",
    "mixed": "No single cell shape dominates.",
}

CYTOPLASM_TEXT = {
    "pale_mucinous": "Pale, faintly blue-grey, abundant cytoplasm, often with a clear vacuole.",
    "dense_eosinophilic": "Dense, opaque pink cytoplasm that fills most of the cell.",
    "clear_vacuolated": "Clear rounded vacuoles within the cells, pushing the nucleus aside.",
    "foamy": "Pale, finely bubbly cytoplasm in large cells inside open spaces.",
    "scant": "Little visible cytoplasm relative to the nucleus.",
}

STROMA_TEXT = {
    "thin_alveolar": "Thin delicate walls between open spaces.",
    "dense_fibrous": "Broad bands or sheets of dense pink collagen.",
    "inflamed": "Abundant small round inflammatory cells between structures.",
    "necrotic": "Structureless pink debris or fragmented nuclei.",
    "minimal": "Little intervening tissue between the epithelial cells.",
}

PROLIFERATION_PATTERN_TEXT = {
    "glandular": (
        "Gland-like or acinar structures with central lumens, columnar cells, or pale mucinous "
        "material between the cells."
    ),
    "solid_squamoid": (
        "Solid islands or sheets of polygonal cells with abundant dense pink cytoplasm, sharp cell "
        "borders, or concentric pink whorls."
    ),
    "neither_clear": "Neither pattern is clearly present, or the field is not a crowded epithelial proliferation.",
}

# Soft maps from a Choice answer to evidence for p0/p1/p2.
CELL_SHAPE_SIGNAL = {
    "columnar_or_cuboidal": (1.0, 0.0, 0.0),
    "polygonal_squamoid": (0.0, 0.0, 1.0),
    "flattened_lining": (0.0, 1.0, 0.0),
    "round_dyscohesive": (0.5, 0.0, 0.5),
    "mixed": (1 / 3, 1 / 3, 1 / 3),
}
CYTOPLASM_SIGNAL = {
    "pale_mucinous": (1.0, 0.0, 0.0),
    "dense_eosinophilic": (0.0, 0.0, 1.0),
    "clear_vacuolated": (0.8, 0.2, 0.0),
    "foamy": (0.0, 1.0, 0.0),
    "scant": (1 / 3, 1 / 3, 1 / 3),
}
STROMA_SIGNAL = {
    "thin_alveolar": (0.0, 1.0, 0.0),
    "dense_fibrous": (0.5, 0.0, 0.5),
    "inflamed": (0.15, 0.7, 0.15),
    "necrotic": (0.2, 0.0, 0.8),
    "minimal": (1 / 3, 1 / 3, 1 / 3),
}
PROLIFERATION_PATTERN_SIGNAL = {
    "glandular": (1.0, 0.0, 0.0),
    "solid_squamoid": (0.0, 0.0, 1.0),
    "neither_clear": (1 / 3, 1 / 3, 1 / 3),
}


def build_questions() -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul

    questions: dict[str, Any] = {}
    for feature in FEATURES:
        questions[feature.id] = Noul(
            instructions=f"{feature.ask}\n\nWhat to look for: {feature.look}",
            criteria=dict(NOUL_CRITERIA),
        )

    questions["profile"] = Choice(
        instructions=(
            "Which description best matches the dominant architecture in `sample`? Read the whole "
            "field first, then pick the description whose pattern occupies the largest area."
        ),
        criteria={profile: PROFILE_TEXT[profile] for profile in PROFILE_IDS},
    )
    # A second pass with a different order and a stricter framing. The two
    # probability vectors are averaged, which reduces option-order bias.
    questions["profile_recheck"] = Choice(
        instructions=(
            "Judge the architecture, not the staining. Which description fits the majority of the "
            "field in `sample`?"
        ),
        criteria={profile: PROFILE_TEXT[profile] for profile in ("p2", "p0", "p1")},
    )
    questions["cell_shape"] = Choice(
        instructions=(
            "Look only at the epithelial cells (the cells that form structures or sheets). What is "
            "their dominant shape in `sample`?"
        ),
        criteria=CELL_SHAPE_TEXT,
    )
    questions["cytoplasm"] = Choice(
        instructions=(
            "Look at the cell body outside the nucleus of the epithelial cells. What is the dominant "
            "cytoplasm in `sample`?"
        ),
        criteria=CYTOPLASM_TEXT,
    )
    questions["stroma"] = Choice(
        instructions=(
            "Look at what sits between the epithelial structures: the supporting tissue and the "
            "background. What lies between them in `sample`?"
        ),
        criteria=STROMA_TEXT,
    )
    questions["proliferation_pattern"] = Choice(
        instructions=(
            "If `sample` contains a crowded epithelial proliferation (many cells growing together "
            "and outgrowing the surrounding tissue), which pattern is the better fit? If it does not, "
            "choose neither_clear."
        ),
        criteria=PROLIFERATION_PATTERN_TEXT,
    )
    return questions


def question_signature() -> str:
    payload = {
        "version": PROTOCOL_VERSION,
        "features": [(feature.id, feature.ask, feature.look) for feature in FEATURES],
        "profiles": PROFILE_TEXT,
        "cell_shape": CELL_SHAPE_TEXT,
        "cytoplasm": CYTOPLASM_TEXT,
        "stroma": STROMA_TEXT,
        "proliferation_pattern": PROLIFERATION_PATTERN_TEXT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def rubric_vector(feature_probabilities: dict[str, float]) -> list[float]:
    """Clinical rubric: sum of present-feature weights, one score per profile."""
    scores = [0.0, 0.0, 0.0]
    for feature in FEATURES:
        value = feature_probabilities.get(feature.id)
        if value is None:
            continue
        present = 1.0 if value >= 0.5 else 0.0
        for index in range(3):
            scores[index] += present * feature.weights[index]
    return scores


def iter_prompt_strings(questions: dict[str, Any]):
    for qid, question in questions.items():
        yield qid
        instructions = getattr(question, "instructions", None)
        if isinstance(instructions, str):
            yield instructions
        criteria = getattr(question, "criteria", None)
        if isinstance(criteria, dict):
            for key, value in criteria.items():
                yield str(key)
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    yield json.dumps(value)


def assert_no_leaks(questions: dict[str, Any]) -> None:
    for text in iter_prompt_strings(questions):
        lowered = text.lower()
        for term in BANNED_PROMPT_TERMS:
            if term in lowered:
                raise ValueError(f"prompt leak: {term!r} in {text!r}")
