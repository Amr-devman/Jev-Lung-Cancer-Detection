"""Variant question banks for the tuning experiment (see ``tune.py``).

Each variant changes the question text or question set of the base protocol and
nothing else. Names are ``vNN_short`` and iterations are appended here as the
experiment progresses; ``note`` records what changed and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VARIANTS: dict[str, dict] = {}

# --------------------------------------------------------------------------
# Iteration 1: baseline — the bank as shipped in jev_cancer/protocol.py
# --------------------------------------------------------------------------
VARIANTS["v00_base"] = {
    "note": "baseline: current protocol (42 Nouls + 6 Choices, described)",
}

# --------------------------------------------------------------------------
# Iterations 2-8: ablations — which question groups carry the accuracy?
# --------------------------------------------------------------------------
VARIANTS["v01_no_adequacy"] = {
    "note": "ablation: drop the 3 adequacy gates (q_*)",
    "drop": ("q_tissue", "q_artifact", "q_stain"),
}
VARIANTS["v02_no_background"] = {
    "note": "ablation: drop the 5 background observations (b_*)",
    "drop": ("b_pigment", "b_rbc", "b_macrophages", "b_inflammation", "b_hemosiderin"),
}
VARIANTS["v03_no_cytology"] = {
    "note": "ablation: drop the 16 cytology observations (c_*)",
    "drop": tuple(f"c_{name}" for name in (
        "dense_cytoplasm", "sharp_borders", "columnar", "vacuoles", "signet", "nucleoli",
        "pleomorphism", "hyperchromasia", "high_nc", "mitoses", "nuclear_contour",
        "chromatin", "hobnail", "polygonal", "shrunken", "cilia",
    )),
}
VARIANTS["v04_no_architecture"] = {
    "note": "ablation: drop the 18 architecture observations (a_*)",
    "drop": tuple(f"a_{name}" for name in (
        "open_spaces", "thin_walls", "architecture_uniform", "glands", "papillae", "mucin",
        "solid", "desmoplasia", "fibrous_bands", "single_file", "necrosis", "surface_columnar",
        "whorl", "spines", "lepidic", "micropapillary", "cribriform", "airspace_filling",
    )),
}
VARIANTS["v05_rubric_only"] = {
    "note": "ablation: no Choices at all — the 42-observation rubric alone",
    "drop": ("profile", "profile_recheck", "cell_shape", "cytoplasm", "stroma",
             "proliferation_pattern"),
}
VARIANTS["v06_choices_only"] = {
    "note": "ablation: no Nouls at all — the 6 Choices alone",
    "drop": tuple(
        f"c_{name}" for name in (
            "dense_cytoplasm", "sharp_borders", "columnar", "vacuoles", "signet", "nucleoli",
            "pleomorphism", "hyperchromasia", "high_nc", "mitoses", "nuclear_contour",
            "chromatin", "hobnail", "polygonal", "shrunken", "cilia",
        )
    ) + tuple(f"a_{name}" for name in (
        "open_spaces", "thin_walls", "architecture_uniform", "glands", "papillae", "mucin",
        "solid", "desmoplasia", "fibrous_bands", "single_file", "necrosis", "surface_columnar",
        "whorl", "spines", "lepidic", "micropapillary", "cribriform", "airspace_filling",
    )) + ("q_tissue", "q_artifact", "q_stain",
          "b_pigment", "b_rbc", "b_macrophages", "b_inflammation", "b_hemosiderin"),
}
VARIANTS["v07_no_recheck"] = {
    "note": "ablation: drop the reordered profile_recheck (order-bias value)",
    "drop": ("profile_recheck",),
}

# --------------------------------------------------------------------------
# Iteration 9: sharpened profile descriptions (error analysis of iter 1)
#
# Both baseline errors are solid-pattern glandular tiles called p2: the profile
# Choice is area-based ("pattern occupying the largest area"), so solid sheets
# win even when gland/mucin tie-breakers are visible. Literature (structured
# clinical prompts, forced-choice framing) says: judge differentiation, and
# demand the discriminating features, not the dominant texture.
# --------------------------------------------------------------------------
P0_SHARP = (
    "Glandular differentiation. Rings of cells around an open central hole (a lumen), lined by "
    "tall cells; pale blue-grey mucin pools between or inside the cells; clear vacuoles that push "
    "nuclei aside; dense pink scar-like tissue between the structures. Even when most of the field "
    "is solid sheets, choose this description if anywhere in the field there are gland lumens, "
    "mucin pools, or tall columnar cells. Judge the type of differentiation, not the largest area."
)
P1_SHARP = (
    "Air-sac architecture. Several open, mostly empty spaces separated by very thin walls one or "
    "two cells thick; little crowded cell growth; scattered dark pigment specks, red blood cells "
    "and pale bubbly scavenger cells inside the spaces. Look for the open airy pattern."
)
P2_SHARP = (
    "Keratinizing solid differentiation. Broad islands or sheets of polygonal cells with abundant "
    "dense opaque pink cytoplasm and sharply drawn cell borders, and at least one keratinizing "
    "tie-breaker: concentric onion-skin whorls of bright pink keratin, fine spine-like bridges "
    "between neighbouring cells, or intensely pink shrunken cells. Solid sheets without a "
    "keratinizing tie-breaker do not fit this description."
)
VARIANTS["v08_sharp_profiles"] = {
    "note": "profiles judge differentiation not area; p2 requires a keratinizing tie-breaker",
    "profile_text": {"p0": P0_SHARP, "p1": P1_SHARP, "p2": P2_SHARP},
    "instructions": {
        "profile": (
            "Which description matches the differentiation shown in `sample`? Look for the "
            "specific tie-breaker features each description names, not just which texture covers "
            "the largest area."
        ),
        "profile_recheck": (
            "Judge the differentiation, not the staining or the largest texture. Which "
            "description fits `sample`?"
        ),
    },
}

# --------------------------------------------------------------------------
# Iteration 10: contrastive forced-choice tie-breaker.
#
# Independent yes/no questions default to agreement (presence-question answer
# bias); a question that names both alternatives and forces a choice uses the
# model's discrimination better. This asks for the differentiation directly.
# --------------------------------------------------------------------------
TIEBREAK_TEXT = {
    "gland_type": (
        "Gland lumens, pale mucin pools, tall columnar cells, or clear vacuoles pushing nuclei "
        "aside are visible. These are the tie-breakers of glandular differentiation."
    ),
    "keratin_type": (
        "Concentric whorls of bright pink keratin, fine spine-like bridges between neighbouring "
        "cells, or dense opaque keratinizing cytoplasm are visible. These are the tie-breakers of "
        "keratinizing differentiation."
    ),
    "both_present": "Clear tie-breakers of both kinds are visible in the field.",
    "neither_clear": "No clear tie-breaker of either kind is visible.",
}
TIEBREAK_SIGNAL = {
    "gland_type": (1.0, 0.0, 0.0),
    "keratin_type": (0.0, 0.0, 1.0),
    "both_present": (0.5, 0.0, 0.5),
    "neither_clear": (1 / 3, 1 / 3, 1 / 3),
}
VARIANTS["v09_tiebreak"] = {
    "note": "add a forced-choice differentiation tie-breaker (gland vs keratin features)",
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.15},
}

# --------------------------------------------------------------------------
# Iteration 11: quantified presence criteria on the noisy tie-breaker Nouls.
#
# Presence questions show a yes-answer bias; requiring >= 2 clear examples
# before answering true raises the evidential bar.
# --------------------------------------------------------------------------
STRICT = {
    "true": (
        "The described property is clearly visible in at least two separate places in the "
        "image. One doubtful example, or an appearance that could be a cutting or staining "
        "artefact, is not enough."
    ),
    "false": (
        "The property is absent, appears at most once, too faint to be sure, or is only a "
        "doubtful hint."
    ),
}
VARIANTS["v10_quantified"] = {
    "note": ">=2 clear examples required for true on the tie-breaker observations",
    "feature_criteria": {fid: STRICT for fid in (
        "a_glands", "a_mucin", "a_whorl", "a_spines", "a_papillae", "a_solid",
        "c_sharp_borders", "c_polygonal", "c_dense_cytoplasm", "a_cribriform",
    )},
}

# --------------------------------------------------------------------------
# Iteration 12: combined v08 + v09 + v10
# --------------------------------------------------------------------------
VARIANTS["v11_combined"] = {
    "note": "sharpened profiles + forced-choice tie-breaker + quantified criteria",
    "profile_text": {"p0": P0_SHARP, "p1": P1_SHARP, "p2": P2_SHARP},
    "instructions": {
        "profile": (
            "Which description matches the differentiation shown in `sample`? Look for the "
            "specific tie-breaker features each description names, not just which texture covers "
            "the largest area."
        ),
        "profile_recheck": (
            "Judge the differentiation, not the staining or the largest texture. Which "
            "description fits `sample`?"
        ),
    },
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.15},
    "feature_criteria": {fid: STRICT for fid in (
        "a_glands", "a_mucin", "a_whorl", "a_spines", "a_papillae", "a_solid",
        "c_sharp_borders", "c_polygonal", "c_dense_cytoplasm", "a_cribriform",
    )},
}

# --------------------------------------------------------------------------
# Iterations 15-18: symmetric differentiation judgment.
#
# Holdout (panel s20261022, iter 13-14) showed v08 overfits: "glands anywhere"
# flips squamous tiles with pseudo-glandular spaces. The fix is contrastive and
# symmetric: p0 requires CLEAR gland evidence AND NO keratinizing tie-breaker;
# p2 requires a keratinizing tie-breaker AND no clear gland evidence.
# --------------------------------------------------------------------------
P0_BALANCED = (
    "Glandular differentiation. Rings of cells around an open central hole (a lumen), lined by "
    "tall cells; pale blue-grey mucin pools between or inside the cells; clear vacuoles that push "
    "nuclei aside; dense pink scar-like tissue between the structures. Choose this description "
    "when gland lumens or mucin pools are clearly present in at least two separate places AND no "
    "keratinizing tie-breaker is present (no concentric pink whorls, no fine bridges between "
    "cells, no dense opaque keratinizing cytoplasm). Solid sheets alone do not qualify."
)
P2_BALANCED = (
    "Keratinizing solid differentiation. Broad islands or sheets of polygonal cells with abundant "
    "dense opaque pink cytoplasm and sharply drawn cell borders, together with at least one "
    "keratinizing tie-breaker: concentric onion-skin whorls of bright pink keratin, fine "
    "spine-like bridges between neighbouring cells, or intensely pink shrunken cells. Open spaces "
    "or lumens that look gland-like do not count against this description unless they are clearly "
    "lined by tall cells in at least two separate places."
)
BALANCED_INSTRUCTIONS = {
    "profile": (
        "Which description matches the differentiation shown in `sample`? Decide by the "
        "tie-breaker features each description names: gland lumens and mucin argue for the "
        "glandular description, whorls and bridges argue for the keratinizing description. Judge "
        "the differentiation, not which texture covers the largest area."
    ),
    "profile_recheck": (
        "Judge the differentiation, not the staining or the largest texture: whorls and bridges "
        "argue for the keratinizing description, clear gland lumens and mucin argue for the "
        "glandular description. Which description fits `sample`?"
    ),
}
VARIANTS["v12_balanced"] = {
    "note": "symmetric tie-break rule: p0 needs clear glands AND no keratin, p2 needs keratin AND no clear glands",
    "profile_text": {"p0": P0_BALANCED, "p1": P1_SHARP, "p2": P2_BALANCED},
    "instructions": BALANCED_INSTRUCTIONS,
}
VARIANTS["v13_balanced_tiebreak"] = {
    "note": "v12 + forced-choice differentiation tie-breaker at low weight (0.08)",
    "profile_text": {"p0": P0_BALANCED, "p1": P1_SHARP, "p2": P2_BALANCED},
    "instructions": BALANCED_INSTRUCTIONS,
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.08},
}
STRICT_KERATIN = {
    "true": (
        "The described property is clearly visible in at least two separate places in the "
        "image. One doubtful example, or an appearance that could be a cutting or staining "
        "artefact, is not enough."
    ),
    "false": (
        "The property is absent, appears at most once, too faint to be sure, or is only a "
        "doubtful hint."
    ),
}
VARIANTS["v14_balanced_strict"] = {
    "note": "v12 + strict >=2 examples on the two hallucination-prone keratin tie-breakers",
    "profile_text": {"p0": P0_BALANCED, "p1": P1_SHARP, "p2": P2_BALANCED},
    "instructions": BALANCED_INSTRUCTIONS,
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}
VARIANTS["v15_balanced_both"] = {
    "note": "v12 + tie-breaker choice (0.08) + strict keratin criteria",
    "profile_text": {"p0": P0_BALANCED, "p1": P1_SHARP, "p2": P2_BALANCED},
    "instructions": BALANCED_INSTRUCTIONS,
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.08},
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}

# --------------------------------------------------------------------------
# Iterations 23-26: consistency-guided profile question.
#
# Evidence mining on panel s20261021: img_013388's tie-breaker answer is
# gland_type (0.59) and a_glands=0.81, yet the holistic profile Choice votes
# p2 (0.60). The Choice contradicts the model's own observations. Structured
# clinical prompting (decision-rule templates) fixes this: give the question an
# explicit ordered rule over the tie-breakers instead of "largest area".
# --------------------------------------------------------------------------
RULE_PROFILE = (
    "Which description matches `sample`? Decide in this order: (1) if concentric bright pink "
    "whorls, fine spine-like bridges between neighbouring cells, or dense opaque pink cytoplasm "
    "in polygonal cells are clearly present, the keratinizing description fits; (2) otherwise, if "
    "rings of cells around open lumens or pale mucin pools are clearly present in at least two "
    "separate places, the glandular description fits; (3) otherwise, if thin-walled open air "
    "spaces dominate the field, the air-sac description fits; (4) otherwise pick the description "
    "closest to the dominant growth pattern."
)
RULE_RECHECK = (
    "Judge the differentiation, not the staining. Decide in this order: (1) clear gland lumens "
    "or mucin pools in at least two separate places favour the glandular description; (2) clear "
    "concentric pink whorls, fine bridges, or dense opaque pink cytoplasm favour the keratinizing "
    "description; (3) thin-walled open air spaces favour the air-sac description; (4) otherwise "
    "pick the description closest to the dominant growth pattern."
)
VARIANTS["v16_rule_profile"] = {
    "note": "profile questions become ordered decision rules over the tie-breakers",
    "instructions": {"profile": RULE_PROFILE, "profile_recheck": RULE_RECHECK},
}
VARIANTS["v17_rule_tiebreak"] = {
    "note": "v16 + forced-choice tie-breaker at 0.08 (consistency between the two questions)",
    "instructions": {"profile": RULE_PROFILE, "profile_recheck": RULE_RECHECK},
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.08},
}
VARIANTS["v18_rule_strict"] = {
    "note": "v16 + strict >=2 examples on the keratin tie-breaker Nouls (a_whorl/a_spines)",
    "instructions": {"profile": RULE_PROFILE, "profile_recheck": RULE_RECHECK},
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}
VARIANTS["v19_rule_all"] = {
    "note": "v16 + tie-breaker choice (0.08) + strict keratin criteria",
    "instructions": {"profile": RULE_PROFILE, "profile_recheck": RULE_RECHECK},
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.08},
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}

# --------------------------------------------------------------------------
# Iterations 31-32: precise p0 criterion.
#
# v08 worked through two mechanisms: the p2 clause (solid sheets need a
# keratinizing tie-breaker) fixed img_000266; the p0 clause ("glands/mucin
# anywhere") fixed img_013388 but flipped squamous img_013445 on mucin alone
# (a_mucin=0.53, a_glands=0.41). Refinement: p0 requires well-formed gland
# rings in >=2 places; mucin alone is not enough. The p2 clause is kept.
# --------------------------------------------------------------------------
P0_V20 = (
    "Glandular differentiation. Rings of cells around an open central hole (a lumen), lined by "
    "tall cells; pale blue-grey mucin pools between or inside the cells; clear vacuoles that push "
    "nuclei aside; dense pink scar-like tissue between the structures. Even when most of the "
    "field is solid sheets, choose this description when well-formed gland rings with open lumens "
    "are clearly present in at least two separate places. Mucin pools alone, or spaces that "
    "merely look gland-like, are not enough. Judge the differentiation, not the largest area."
)
VARIANTS["v20_precise_glands"] = {
    "note": "v08 split apart: p2 keeps the keratin-tie-breaker clause; p0 needs gland rings in >=2 places, not mucin alone",
    "profile_text": {"p0": P0_V20, "p1": P1_SHARP, "p2": P2_SHARP},
    "instructions": {
        "profile": (
            "Which description matches the differentiation shown in `sample`? Look for the "
            "specific tie-breaker features each description names, not just which texture covers "
            "the largest area."
        ),
        "profile_recheck": (
            "Judge the differentiation, not the staining or the largest texture. Which "
            "description fits `sample`?"
        ),
    },
}
VARIANTS["v21_precise_strict"] = {
    "note": "v20 + strict >=2-example criteria on the keratin tie-breakers (a_whorl/a_spines)",
    "profile_text": {"p0": P0_V20, "p1": P1_SHARP, "p2": P2_SHARP},
    "instructions": {
        "profile": (
            "Which description matches the differentiation shown in `sample`? Look for the "
            "specific tie-breaker features each description names, not just which texture covers "
            "the largest area."
        ),
        "profile_recheck": (
            "Judge the differentiation, not the staining or the largest texture. Which "
            "description fits `sample`?"
        ),
    },
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}

# --------------------------------------------------------------------------
# Iterations 35-38: carry the tie-breaker evidence at real weight.
#
# The tie-breaker Choice separates the remaining cases cleanly
# (gland_type 0.59/0.76 on the adeno tiles, neither_clear 0.58 on the
# squamous flip case), so its mapped signal is added on top of v21.
# --------------------------------------------------------------------------
def _v21plus(name, note, weight):
    return {
        "note": note,
        "profile_text": {"p0": P0_V20, "p1": P1_SHARP, "p2": P2_SHARP},
        "instructions": {
            "profile": (
                "Which description matches the differentiation shown in `sample`? Look for the "
                "specific tie-breaker features each description names, not just which texture "
                "covers the largest area."
            ),
            "profile_recheck": (
                "Judge the differentiation, not the staining or the largest texture. Which "
                "description fits `sample`?"
            ),
        },
        "extra_choices": [(
            "differentiation_tiebreak",
            (
                "When `sample` shows both gland-like structures and solid sheets, decide which "
                "differentiation the cells are showing. Look for the specific tie-breakers named "
                "in each option, not the largest texture."
            ),
            TIEBREAK_TEXT,
        )],
        "extra_signals": {
            "tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}
        },
        "weights": {"tiebreak": weight},
        "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
    }


VARIANTS["v22_tiebreak_015"] = _v21plus(
    "v22_tiebreak_015",
    "v21 + differentiation tie-breaker carried at weight 0.15",
    0.15,
)
VARIANTS["v23_tiebreak_025"] = _v21plus(
    "v23_tiebreak_025",
    "v21 + differentiation tie-breaker carried at weight 0.25",
    0.25,
)

# --------------------------------------------------------------------------
# Iterations 39-42: self-referential conditioning.
#
# The two remaining knife-edge tiles differ mainly in a_glands (0.80 adeno vs
# 0.41 squamous); every textual clause that fires for one fires for the other.
# Question-decomposition literature suggests conditioning the final question on
# the model's own earlier observations: the profile question now decides FROM
# the answers already given, so the a_glands judgment is reused instead of
# re-derived under a different framing.
# --------------------------------------------------------------------------
SELF_PROFILE = (
    "You already described `sample` in the questions above. Decide from those observations. If "
    "gland-like structures with central open lumens were present, the glandular description fits, "
    "unless whorls or fine bridges were also clearly present. If open air spaces with thin walls "
    "dominated and no crowded growth was seen, the air-sac description fits. Otherwise, if solid "
    "sheets of cells without gland lumens were seen, the keratinizing description fits."
)
SELF_RECHECK = (
    "Decide again from the observations you made, ignoring staining: clear gland lumens argue for "
    "the glandular description, clear whorls or bridges argue for the keratinizing description, "
    "thin-walled open air spaces argue for the air-sac description. Which description fits `sample`?"
)
VARIANTS["v24_self_conditioned"] = {
    "note": "profile questions decide from the model's own observation answers instead of re-reading the image",
    "instructions": {"profile": SELF_PROFILE, "profile_recheck": SELF_RECHECK},
    "profile_text": {"p0": P0_V20, "p1": P1_SHARP, "p2": P2_SHARP},
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}
VARIANTS["v25_self_tiebreak"] = {
    "note": "v24 + differentiation tie-breaker at 0.15",
    "instructions": {"profile": SELF_PROFILE, "profile_recheck": SELF_RECHECK},
    "profile_text": {"p0": P0_V20, "p1": P1_SHARP, "p2": P2_SHARP},
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
    "extra_choices": [(
        "differentiation_tiebreak",
        (
            "When `sample` shows both gland-like structures and solid sheets, decide which "
            "differentiation the cells are showing. Look for the specific tie-breakers named in "
            "each option, not the largest texture."
        ),
        TIEBREAK_TEXT,
    )],
    "extra_signals": {"tiebreak": {"question": "differentiation_tiebreak", "mapping": TIEBREAK_SIGNAL}},
    "weights": {"tiebreak": 0.15},
}

# --------------------------------------------------------------------------
# Iteration 46: v21 + a narrow mucin concession.
#
# On 99 validation images v08's loose p0 clause gained adeno (0.88 vs 0.85)
# but lost squamous (0.79 vs 0.88). v26 keeps v21's structure and lets a
# definite mucin pool qualify only when paired with gland evidence.
# --------------------------------------------------------------------------
P0_V26 = (
    "Glandular differentiation. Rings of cells around an open central hole (a lumen), lined by "
    "tall cells; pale blue-grey mucin pools between or inside the cells; clear vacuoles that push "
    "nuclei aside; dense pink scar-like tissue between the structures. Even when most of the "
    "field is solid sheets, choose this description when well-formed gland rings with open lumens "
    "are clearly present in at least two separate places, or when a definite pale mucin pool sits "
    "together with at least one gland ring or clear vacuolated columnar cells. Mucin alone, or "
    "spaces that merely look gland-like, is not enough. Judge the differentiation, not the "
    "largest area."
)
VARIANTS["v26_mucin_concession"] = {
    "note": "v21 + mucin pool qualifies only alongside gland evidence",
    "profile_text": {"p0": P0_V26, "p1": P1_SHARP, "p2": P2_SHARP},
    "instructions": {
        "profile": (
            "Which description matches the differentiation shown in `sample`? Look for the "
            "specific tie-breaker features each description names, not just which texture covers "
            "the largest area."
        ),
        "profile_recheck": (
            "Judge the differentiation, not the staining or the largest texture. Which "
            "description fits `sample`?"
        ),
    },
    "feature_criteria": {"a_whorl": STRICT_KERATIN, "a_spines": STRICT_KERATIN},
}
