# What a lung biopsy shows: the clinical basis for the Jev question bank

This note is the domain reference behind `jev_cancer/protocol.py`. It explains
what each entity is, what a pathologist looks for on an H&E slide, which findings
separate the entities, and which findings are unreliable on a small biopsy. It is
written for people building the experiment, not for diagnosis. Nothing here is
medical advice.

The dataset in this repository is **LungHist700**
(`abdullahhasansajjad/lunghist700` on Kaggle), 691 H&E fields of lung tissue in
three class folders under `LungHist700_combined/data/images`:

| Folder in the download | Images | Entity |
|---|---:|---|
| `aca` | 280 | lung adenocarcinoma |
| `nor` | 151 | normal lung tissue |
| `ssc` | 260 | lung squamous cell carcinoma |

The download also ships a `data.csv` with per-image subclass (differentiation
grade), magnification (20×/40×) and patient id. The pipeline never reads it —
grading is on the three superclass folders only. All images are 1600 × 1200 px,
RGB, JPEG. They are image fields, not whole slides: there is no slide-level
architecture, no stain metadata, no IHC, and no patient context beyond what the
unused CSV holds.

## 1. Why the H&E image matters, and what it cannot show

H&E (haematoxylin and eosin) stains nuclei blue-purple and cytoplasm,
collagen and keratin pink. On a tile, a pathologist can judge:

- **architecture** — glands, papillae, solid nests, alveolar septa, stroma;
- **cytology** — cell size and shape, cytoplasm, nuclear contour, nucleoli,
  chromatin, nuclear-to-cytoplasmic ratio, mitoses;
- **background** — necrosis, pigment, inflammation, blood, mucin.

H&E cannot show the protein markers that resolve the hardest cases. The standard
minimal immunohistochemistry panel for non-small-cell lung cancer is **TTF-1**
(adenocarcinoma, ~80–90% sensitive) and **p40** (squamous, highly specific,
~100% in several series); Napsin A and CK5/6 are common additions. Molecular
testing (EGFR, ALK, ROS1, BRAF, KRAS, MET, RET, NTRK, HER2) is also invisible on
H&E.

The classifier in this repo therefore uses **morphology only**. That is a real
constraint, not a shortcut: it is the same information a pathologist has before
ordering stains, and it is exactly the situation where a triage model is useful.

A second constraint is scale. A 20×/40× field covers a small area. Features such
as "invasion of the pleura" or "spread through air spaces" cannot be judged from
one field. The model is asked only about properties that are visible in the
field.

## 2. Lung adenocarcinoma

### Clinical picture

Adenocarcinoma is the most common histological type of lung cancer (~40% of
NSCLC, and the most common type in never-smokers, women and younger patients).
It is typically **peripheral**, may present as a ground-glass nodule, and is
often discovered incidentally. It arises from glandular epithelium and shows
glandular differentiation, mucin production, or pneumocyte marker expression
(WHO 2021). Precursor lesions are atypical adenomatous hyperplasia (AAH),
adenocarcinoma in situ (AIS) and minimally invasive adenocarcinoma (MIA).

### Architecture

The 2015 WHO/IASLC classification recognises five major patterns, which are
combined in one tumour:

| Pattern | What it looks like |
|---|---|
| **Lepidic** | Tumour cells grow along pre-existing alveolar septa; the alveolar architecture is preserved |
| **Acinar** | Glands or acini with central lumens, often with mucin |
| **Papillary** | Finger-like fronds with a fibrovascular core covered by tumour cells |
| **Micropapillary** | Small tufts of tumour cells without a fibrovascular core, floating in alveolar spaces |
| **Solid** | Sheets of polygonal cells without glands; mucin may be present |

The IASLC 2020 grading system uses the predominant pattern: grade 1 =
lepidic, grade 2 = acinar or papillary, grade 3 = solid, micropapillary or
complex glands. Grade correlates with prognosis.

### Cytology

- Columnar or cuboidal cells, often with apical mucin.
- Pale, vacuolated or mucinous cytoplasm; signet-ring cells in some tumours.
- Enlarged nuclei, irregular nuclear contours, prominent nucleoli.
- Hobnail cells (nucleus at the luminal tip) in some patterns.
- Intracytoplasmic mucin displacing the nucleus.

### Stroma and background

- Desmoplastic (dense pink fibrous) stroma between infiltrating glands.
- Single-file or small-cluster infiltration at the tumour edge.
- Mucin pools, sometimes with floating tumour cells.
- Necrosis is less common than in squamous cell carcinoma.

### Immunohistochemistry and molecular biology (not visible on H&E)

TTF-1 (NKX2-1) positive in ~80–90%, Napsin A positive in a similar range,
CK7 positive; p40/p63 negative. Actionable drivers are concentrated here:
EGFR mutations, ALK and ROS1 rearrangements, BRAF V600E, MET exon 14 skipping,
RET and NTRK fusions, HER2 alterations. This is why separating adenocarcinoma
from squamous carcinoma matters: the treatment pathways differ.

## 3. Lung squamous cell carcinoma

### Clinical picture

Squamous cell carcinoma accounts for roughly 20–30% of NSCLC and is strongly
associated with smoking. It typically arises **centrally** from the bronchial
epithelium, may cavitate, and can present with haemoptysis or obstructive
pneumonia. Precursor lesions are squamous metaplasia, dysplasia and carcinoma in
situ.

### Architecture

- Cohesive **nests, islands or sheets** of polygonal cells.
- **Keratinization**: intracellular keratin, keratin pearls (concentric whorls
  of brightly eosinophilic keratin), and dyskeratotic cells.
- **Intercellular bridges** (desmosomal attachments) between cells.
- Central necrosis within nests ("comedo" pattern) is common.
- Desmoplastic stroma around infiltrating nests.

In a 2025 series of 56 resected SCCs, keratin pearls were present in 78.9% and
intercellular bridges in 75.4% on H&E — useful, but neither is universal, and
both are rarer in poorly differentiated tumours.

### Cytology

- Abundant, dense, opaque eosinophilic cytoplasm.
- Sharply defined cell borders; flattened or polygonal cell shape.
- Enlarged, hyperchromatic nuclei with coarse chromatin.
- Prominent nucleoli in some tumours; high mitotic rate.
- Dyskeratotic cells: shrunken, intensely pink cytoplasm with a condensed
  nucleus.

### Immunohistochemistry and molecular biology (not visible on H&E)

p40 (ΔNp63) is the most specific marker, with p63 and CK5/6 also used; TTF-1 and
Napsin A are negative. There is no routine targeted therapy comparable to the
adenocarcinoma drivers; immunotherapy (PD-L1) is central. FGFR1 amplification,
SOX2 amplification and PIK3CA mutations are common.

## 4. Normal / benign lung tissue

### Normal histology

- Alveolar spaces lined by flattened type I pneumocytes, with cuboidal type II
  pneumocytes at corners.
- Thin alveolar septa (one or two cell layers) containing capillaries.
- Alveolar macrophages, sometimes with anthracotic (black) pigment.
- No cellular crowding, no nuclear atypia, no infiltrative glands, no
  desmoplasia, no necrosis.

### Reactive mimics that matter

A small biopsy can show reactive changes that look alarming:

| Mimic | Why it is tricky | What argues against malignancy |
|---|---|---|
| Reactive type II pneumocyte hyperplasia | Enlarged pneumocytes with nucleoli and mitoses | Preserved alveolar architecture, no desmoplasia, no invasive glands |
| Organising pneumonia | Fibroblastic plugs fill alveoli | Uniform plugs, no cytologic atypia, no destructive growth |
| Squamous metaplasia | Squamous-looking cells in airways | No atypia, no invasion, no keratin pearls |
| Alveolar haemorrhage | Blood fills spaces, macrophages | Clinical context, no tumour cells |
| Anthracosis / inflammation | Dense pigment and lymphocytes | Pigment is extracellular/within macrophages; no epithelial atypia |

This is why the normal class is not simply "empty tissue": the questions have to
distinguish **orderly architecture** from **crowded, atypical, infiltrative
growth**, not just "cells present vs absent".

## 5. Large cell carcinoma (not in this mirror)

Large cell carcinoma is a diagnosis of exclusion: a poorly differentiated
non-small-cell carcinoma with no glandular, squamous or neuroendocrine
differentiation, usually requiring a resection specimen and IHC. It is not
present in this dataset. If a future split adds it, the protocol needs a fourth
profile and a "cannot determine on H&E" option, because the morphology overlaps
with solid adenocarcinoma and poorly differentiated squamous carcinoma.

## 6. What actually separates the three classes in this dataset

The discriminators below are the ones a pathologist would use on a single H&E
tile, ranked by how reliably they appear in this image collection.

| Observation | Adenocarcinoma | Squamous cell carcinoma | Normal lung |
|---|---|---|---|
| Open alveolar air spaces | Rare (except lepidic) | Rare | **Characteristic** |
| Thin alveolar septa | Rare | Rare | **Characteristic** |
| Gland/acinar lumens | **Characteristic** | Rare | Rare |
| Papillary fronds | Common | Rare | Rare |
| Mucin / pale pools | Common | Rare | Rare |
| Solid cohesive nests | Sometimes (solid pattern) | **Characteristic** | No |
| Keratin whorls | No | **Characteristic** | No |
| Intercellular bridges | No | Common | No |
| Dense opaque pink cytoplasm | Uncommon | **Characteristic** | No |
| Sharp cell borders | Uncommon | **Characteristic** | No |
| Columnar cells | **Characteristic** | Rare | Sometimes (airway) |
| Clear vacuoles / signet cells | Common | Rare | No |
| Prominent nucleoli | Common | Common | Reactive only |
| Nuclear pleomorphism | Common | Common | Absent/mild |
| Desmoplastic stroma | Common | Common | No |
| Central necrosis in nests | Sometimes | Common | No |
| Anthracotic pigment | Uncommon | Uncommon | **Common** |
| Alveolar macrophages | Uncommon | Uncommon | **Common** |

The hard boundary is **solid-pattern adenocarcinoma vs poorly differentiated
squamous carcinoma**. Both can be sheets of atypical cells with necrosis and
desmoplasia. On H&E alone, keratinization, intercellular bridges and mucin are
the tie-breakers; when none is present, a pathologist orders p40 and TTF-1.
The model has no such option, so the protocol must:

1. ask about the tie-breakers explicitly,
2. keep a confidence signal and abstain rather than force a label,
3. never present the answer as a diagnosis.

That is why the pipeline reports a probability distribution and a confidence
curve, not just an argmax.

## 7. From clinical features to questions Jev can answer

Four rules shaped `protocol.py`:

1. **One observation per question.** "Is there keratin and are there bridges?"
   is two questions. Atomic Nouls are easier to calibrate and easier to audit.
2. **Describe, do not diagnose.** A question says "a whorl of concentric pink
   material", not "a keratin pearl", and never "squamous cell carcinoma". The
   mapping from observation to entity lives in code, where it can be reviewed.
3. **Say what to look for.** Every question is followed by a plain-language
   description of what the observation means and what it looks like in pixels —
   the same words a pathologist would use to teach a student. Vocabulary the
   model cannot ground visually is a coin flip; the description turns each term
   into a visual task.
4. **Ask about what a single field of view can show.** Architecture, cytoplasm,
   nuclei, stroma and background — not invasion, metastasis, or molecular
   status.

The full bank is documented in [`question_bank.md`](question_bank.md).

## 8. Limitations

- LungHist700 is small (691 images) and not patient-independent: `data.csv`
  shows the same tissue field at both 20× and 40× under one image id, and
  images are grouped by patient. Image-level confidence intervals overstate
  independence.
- The dataset labels are the source dataset's labels. They are not re-reviewed
  by a pathologist for this experiment, and the differentiation-grade subclass
  labels in `data.csv` are unused.
- The normal class is normal lung tissue, not a set of benign mimics. A model
  that learns "alveolar architecture = normal" will not necessarily recognise
  reactive atypia in a real biopsy.
- H&E appearance varies with scanner, stain batch and fixation. A model tuned on
  one collection may not transfer.
- A 20×/40× field has no slide-level context. Any clinical claim would need
  validation on whole slides, multiple institutions, and a pathologist read.

## References

- WHO Classification of Thoracic Tumours, 5th edition (2021).
- Travis WD et al. The 2015 World Health Organization Classification of Lung
  Tumors. *J Thorac Oncol.* 2015.
- Moreira AL et al. A grading system for invasive pulmonary adenocarcinoma.
  *J Thorac Oncol.* 2020.
- Nicholson AG et al. The 2021 WHO Classification of Lung Tumors: impact of
  advances since 2015. *J Thorac Oncol.* 2022.
- PathologyOutlines: lung adenocarcinoma, squamous cell carcinoma.
- Patel S et al. Evaluation of TTF-1, Napsin A, p40 and p63 in subtyping NSCLC.
  PMC12308192 (2025).
- NCCN Guidelines: Non-Small Cell Lung Cancer (current version at time of
  writing).
