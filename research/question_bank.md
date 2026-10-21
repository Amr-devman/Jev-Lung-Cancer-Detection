# The Jev question bank

`jev_cancer/protocol.py` is the executable version of this document. It contains
42 yes/no observations and 6 multiple-choice questions. Every question is
followed by a plain-language **what to look for** description, so the model is
told what the observation means and what it looks like in pixels before it
answers. The prompts name morphology only. They never name a diagnosis, a
dataset folder, or a filename.

## Design rules

1. **Atomic.** One question asks about one observable property. If a question
   needs "and" or "or", it becomes two questions.
2. **Described, not just asked.** Every observation carries a plain-language
   description of what it means and what to look for ("rings of cells around a
   central open hole … circles, ovals, crooked rings, crescents or short
   tubes"). A question the model cannot ground in pixels is a coin flip; the
   description is what turns vocabulary into a visual task.
3. **Descriptive, not diagnostic.** "A whorl of concentric pink material", not
   "a keratin pearl". "Open air spaces with thin walls", not "normal lung". The
   mapping from observation to entity lives in code, where it can be audited.
4. **Visible in a single field of view.** No questions about invasion,
   metastasis, pleural spread, or molecular status.
5. **Balanced polarity.** Every question is phrased so that "yes" means the
   property is present. Each Noul carries shared outcome criteria: `true` = the
   property is clearly visible, `false` = absent, too faint, or a doubtful hint.
6. **One image per request.** No reference images and no labels are ever in the
   state; the model sees the tile and the question bank, nothing else.

## Neutral profiles

The model is never asked "which diagnosis is this?". It is asked which of three
architecture descriptions matches the field. The profile names seen by the model
are `p0`, `p1`, `p2`; the mapping to diagnoses is in `config.PROFILE_TO_CLASS`
and is used only by the grader.

| Profile | Description shown to the model | Morphologic family |
|---|---|---|
| `p0` | Glandular growth. Rounded or irregular rings of cells around an open central hole (a lumen), lined by tall cells; pale blue-grey mucin pools between or inside the structures; dense pink scar-like tissue separates the structures. | glandular / mucinous |
| `p1` | Air-sac architecture. Several open, mostly empty spaces separated by very thin walls one or two cells thick; little crowded cell growth; scattered dark pigment specks, red blood cells and pale bubbly scavenger cells inside the spaces. | alveolar |
| `p2` | Solid cohesive growth. Broad islands or sheets of polygonal cells with abundant dense opaque pink cytoplasm and sharply drawn cell borders; concentric onion-skin whorls of bright pink material; dead pink debris in the centre of the islands. | solid cohesive |

The architecture Choice is asked twice: once in the order above, once reordered
and framed as "judge the architecture, not the staining". The two probability
vectors are averaged, which reduces option-order bias.

## Observation table

Weights are the transparent clinical rubric: how much a "yes" supports each
profile. `+++` strong evidence for, `++` moderate, `+` weak, `–` neutral, `−`
weak evidence against, `−−` moderate evidence against, `−−−` strong evidence
against.

| id | Question (the field = `sample`) | What to look for | p0 | p1 | p2 |
|---|---|---|:--:|:--:|:--:|
| `q_tissue` | Does the field contain intact tissue with visible cells and extracellular structure? | Real tissue: groups of cells with visible dark nuclei plus the pink material between them. Blank glass, blur, or specks do not count. | – | – | – |
| `q_artifact` | Is more than half of the field covered by blood, crush artefact, or structureless debris? | Smears of red blood, tissue crushed into streaks, or shapeless debris without recognisable cells. | – | – | – |
| `q_stain` | Are nuclei and cytoplasm distinguishable across most of the field? | Nuclei read as dark blue-purple dots and cell bodies as pink; too pale, too dark or washed out means no. | – | – | – |
| `a_open_spaces` | Does the field contain multiple open, air-filled spaces bounded by thin tissue walls? | Empty-looking round or irregular holes like the air sacs of healthy lung, mostly white inside, not packed with cells or fluid. | – | +++ | −− |
| `a_thin_walls` | Are the walls between those open spaces thin, roughly one or two cells thick? | A very fine line of pink tissue, one or two cells across, with small dark nuclei spaced along it. | – | +++ | − |
| `a_architecture_uniform` | Is the tissue architecture orderly and similar across most of the field? | A repeating, calm, orderly layout with similar structures side by side, versus a chaotic mixed one. | − | ++ | − |
| `a_glands` | Does the field contain rounded or irregular gland-like structures with a central open lumen? | Rings of cells around a central open hole: circles, ovals, crooked rings, crescents or short tubes depending on the cut. | +++ | −− | − |
| `a_papillae` | Does the field contain finger-like projections of tissue covered by a layer of cells? | Finger- or tree-like projections into an open space: a thin pink core covered by a tidy layer of cells. | ++ | − | − |
| `a_mucin` | Does the field contain pools of pale, homogeneous, slightly blue-grey material between cells? | Lakes of pale, smooth, slightly blue-grey slimy material, uniform and glassy, without cells inside. | +++ | − | − |
| `a_solid` | Does the field contain broad solid islands or sheets of cells with no open lumen inside them? | Large solid sheets or islands of packed cells with no open hole in the middle. | + | −− | +++ |
| `a_desmoplasia` | Is there dense pink fibrous tissue separating groups of cells? | Dense pink scar-like fibrous tissue between cell groups, brighter pink than the cell bodies. | + | −− | + |
| `a_fibrous_bands` | Are there broad bands of dense pink collagen dividing the tissue into compartments? | Wide ribbons of dense pink collagen cutting the tissue into compartments. | + | − | + |
| `a_single_file` | Do cells run in single-file lines through fibrous tissue? | Cells marching one behind the other through fibrous tissue, like beads on a string. | ++ | − | − |
| `a_necrosis` | Do the centres of cell islands contain structureless pink debris or fragmented nuclei? | Dead tissue at the centre of large cell groups: shapeless pink debris or broken dark nuclear fragments. | + | − | ++ |
| `a_surface_columnar` | Is there a surface lining of tall cells whose nuclei sit at different heights? | Tall narrow cells side by side like a picket fence with staggered nuclear heights. | + | + | − |
| `a_whorl` | Does the field contain a whorl of concentric, brightly pink material inside a group of cells? | Onion-skin layers of bright pink material wrapped around a centre inside a cell group. | − | − | +++ |
| `a_spines` | Are fine spine-like connections visible between neighbouring cells? | Fine spine-like lines bridging the narrow gaps between cells in a solid group. | − | − | +++ |
| `a_lepidic` | Do cells line existing air spaces in a single layer without filling the space? | A single neat layer of cells along the walls of air spaces, like wallpaper, without filling them. | ++ | + | − |
| `a_micropapillary` | Are there small tufts of cells floating in open spaces without a fibrous core? | Small free-floating tufts of cells inside open spaces, with no supporting core. | ++ | − | – |
| `a_cribriform` | Are there sieve-like spaces within a solid group of cells? | A solid cell group riddled with small round holes, like a sieve or swiss cheese. | ++ | − | – |
| `a_airspace_filling` | Are open spaces filled by sheets of cells rather than by air or macrophages? | Open spaces packed with sheets of cells, versus empty or holding pale scavenger cells. | + | − | + |
| `c_dense_cytoplasm` | Do most cells have abundant, dense, opaque pink cytoplasm? | Cell bodies that are dense opaque pink, so cells look plump and solid rather than mostly nucleus. | − | − | +++ |
| `c_sharp_borders` | Do neighbouring cells have sharply defined borders rather than blending together? | Clearly drawn cell boundaries, each cell a separate tile, not a merged pink mass. | – | − | +++ |
| `c_columnar` | Are most cells taller than they are wide? | Column-shaped cells, taller than wide, with the nucleus toward the base. | ++ | − | − |
| `c_vacuoles` | Do cells contain clear rounded vacuoles that push the nucleus to one side? | Clear round bubbles in the cell body displacing the dark nucleus. | ++ | – | − |
| `c_signet` | Do cells contain a large pale vacuole that pushes the nucleus to the cell rim? | One large pale bubble filling the cell and squashing the nucleus flat against the rim. | ++ | – | − |
| `c_nucleoli` | Do many cells have enlarged, prominent nucleoli? | A clearly visible large dark spot inside many nuclei. | + | – | + |
| `c_pleomorphism` | Do nuclei vary markedly in size and shape from one cell to the next? | Nuclei differing a lot cell to cell: some large and lumpy, some small and round. | + | – | + |
| `c_hyperchromasia` | Are many nuclei intensely dark? | Nuclei that are intensely dark, almost black-purple, like ink dots. | + | – | + |
| `c_high_nc` | Do many cells have a nucleus that fills most of the cell? | Nuclei filling nearly the whole cell, leaving only a thin rim of pink. | + | – | + |
| `c_mitoses` | Are mitotic figures visible? | Dark chromatin clumps shaped like starbursts, fuzzy spiders or split blobs instead of smooth nuclei. | + | – | + |
| `c_nuclear_contour` | Do many nuclei have irregular, angular or folded contours? | Nuclear outlines that are irregular, angular, notched or folded, not smooth circles. | + | – | + |
| `c_chromatin` | Is the chromatin coarse and clumped in many nuclei? | Coarse, clumped nuclear texture like grains of dark sand, not fine and even. | + | – | + |
| `c_hobnail` | Do some cells bulge into an open space with the nucleus at the outer tip? | Cells bulging into an open space like a thumb into a balloon, nucleus at the outer tip. | ++ | – | − |
| `c_polygonal` | Are most cells flattened or polygonal with angular outlines? | Angular, polygonal or flattened cells with straight-ish edges, like floor tiles. | − | – | ++ |
| `c_shrunken` | Are there individual cells with shrunken, intensely pink cytoplasm and a small dark nucleus? | Single dead shrunken cells: intensely pink dense cytoplasm, tiny condensed dark nucleus. | − | − | +++ |
| `c_cilia` | Are there hair-like cilia on the surface of the cells? | Fine hair-like lines sticking out of the cell surface like a tiny brush border. | − | ++ | − |
| `b_pigment` | Is there black or dark brown granular pigment in the tissue? | Dark black or brown granular specks like soot, scattered in tissue or inside cells. | – | ++ | – |
| `b_rbc` | Are red blood cells present outside blood vessels or within open spaces? | Tiny uniform red-orange discs outside vessels or loose in open spaces. | – | + | – |
| `b_macrophages` | Are there large cells with pale foamy cytoplasm sitting inside open spaces? | Large rounded cells with pale bubbly cytoplasm and small dark nuclei inside open spaces. | – | + | – |
| `b_inflammation` | Is there a dense crowd of small round inflammatory cells? | Dense crowds of small round cells with almost no cytoplasm and dark round nuclei. | – | + | – |
| `b_hemosiderin` | Is there golden-brown granular pigment within large cells? | Coarse golden-brown granules inside large cells, left by old bleeding. | – | + | – |

### Why these observations

**Adequacy (`q_*`).** A tile can be mostly blood, crush artefact or necrosis. The
three adequacy questions do not vote for a class; they tell the reader how much
to trust the rest and are the first thing a pathologist checks.

**Architecture (`a_*`).** Architecture is the most reliable signal on a tile and
the basis of the WHO classification. Open air spaces with thin walls are the
defining feature of preserved alveolar parenchyma; glands, papillae and mucin
define glandular differentiation; solid islands, whorled pink material and
spine-like connections define squamous differentiation. Desmoplasia and
single-file infiltration are shared invasion cues. Central necrosis is more
common in solid nests.

**Cytology (`c_*`).** Cytology separates the two carcinoma profiles when
architecture is ambiguous: columnar cells with vacuoles and hobnail shapes point
to glandular differentiation; abundant dense cytoplasm, sharp borders, polygonal
shape and shrunken pink cells point to squamous differentiation. Pleomorphism,
hyperchromasia, high N:C ratio and mitoses are shared malignancy cues, so they
carry the same small weight for both carcinoma profiles and none for the
alveolar profile.

**Background (`b_*`).** Pigment, red cells, foamy macrophages and inflammation
are typical of preserved lung tissue and are common in the normal field. They are
weak evidence, not proof: a tumour can contain all of them.

### Deliberately absent

- **Diagnosis names.** No prompt contains "adenocarcinoma", "squamous",
  "carcinoma", "normal", "malignant" or "tumour". `protocol.assert_no_leaks`
  enforces this before every request.
- **Box questions.** Boxes were dropped entirely: they never carried class
  evidence, and a Box answer always returns four edges even when the marker is
  absent, which invited fabricated annotations and extra post-processing.
- **IHC and molecular markers.** TTF-1, Napsin A, p40, p63, CK5/6, EGFR, ALK and
  the rest cannot be seen on H&E. Asking about them would invite hallucination.
- **Staging and invasion.** A tile cannot show pleural invasion, vascular
  invasion, or spread through air spaces reliably.
- **Patient metadata.** None is available, and none is invented.

## Multiple-choice questions

Beside the architecture Choice (asked twice), three descriptive Choices and one
growth-pattern Choice feed the fusion as soft evidence:

| Choice | Question | Options |
|---|---|---|
| `cell_shape` | What is the dominant shape of the epithelial cells? | columnar/cuboidal, polygonal/squamoid, flattened lining, round/dyscohesive, mixed |
| `cytoplasm` | What is the dominant cytoplasm of the epithelial cells? | pale mucinous, dense eosinophilic, clear/vacuolated, foamy, scant |
| `stroma` | What lies between the epithelial structures? | thin alveolar walls, dense fibrous, inflamed, necrotic, minimal |
| `proliferation_pattern` | If there is a crowded epithelial proliferation, which pattern fits? | glandular, solid/squamoid, neither clear |

Each option is itself a short description ("pale, faintly blue-grey, abundant
cytoplasm, often with a clear vacuole"), not a bare label, so the model compares
descriptions rather than decoding category names.

## How answers become a class

Seven signals are computed independently from the one response:

- `profile` and `profile_recheck`: the two architecture Choice probability
  vectors;
- `proliferation_pattern`, `cell_shape`, `cytoplasm`, `stroma`: each Choice
  probability vector mapped through a fixed soft table onto p0/p1/p2;
- `rubric`: the weighted 42-observation vector, passed through a softmax.

They are combined by a weighted mean with fixed clinical-prior weights
(`fusion.WEIGHTS`). The result is a probability for `p0`, `p1`, `p2` plus a
confidence and a margin. The grader reports the full distribution and a
confidence/coverage curve rather than pretending every tile is decidable.
