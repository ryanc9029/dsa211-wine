# Declaration of AI tool use

## Tools used

- Claude / Claude Code (Anthropic), used for code assistance and review.
- No other AI tools were used.

## How the tools were used

[TODO: describe, in your own words, which parts of the project the tools were used for (for example: which notebooks or functions, refactoring, debugging, reviewing analysis choices, drafting markdown) and how much of the output was used as-is versus modified.]

### Record of AI-assisted changes in revision v2 (factual; for the group to check and edit)

The group specified the following changes, and Claude Code (Anthropic) implemented them in code and markdown and ran the notebooks:
- removing repeat rows (one row per wine) and switching to `StratifiedKFold`;
- a single one-SE selection rule (`wine_utils.one_se_choice`) used for every CV choice;
- per-model best-subset selection with nested-CV evaluation, the all-11 reference and the accuracy-simplicity trade-off graph;
- removing the test-set audit and renaming notebook 05 to `05_test_and_refit.ipynb`;
- the quality-cut-off justification;
- an error analysis in notebook 04;
- `results/NUMBERS_FOR_SLIDES.md` and `results/CHANGES.md`.

Claude Code also drafted the markdown explanations in the notebooks and the README. An attempt to check the wording of Cortez et al. (2009) online failed, so no claim about that paper's cut-off was made.

### Record of AI-assisted changes after revision v2 (factual; for the group to check and edit)

The group asked Claude Code to check the draft report against the repository. Claude Code compared the report's figures and claims with the notebooks and `results/`, pointed out inaccurate, unsupported or missing statements (for example the description of nested cross-validation), and suggested rewordings, which the group reviewed and edited. At the group's request, Claude Code then made these repository changes:
- stated the definition of a good wine at the top of the split notebook;
- added a cell to the EDA notebook printing each predictor's median for good and not-good wines, and added chlorides and the class overlap to the box-plot commentary, so that the figures quoted in report sections 4.1 and 4.2 trace to notebook outputs;
- quoted the share of good wines as 20.8% throughout, and compared white and red wine on unique wines only;
- renamed `wine_white_data_split.ipynb` and `EDA.ipynb` to `01_data_split.ipynb` and `02_eda.ipynb`, updated every reference, and removed the undefined workflow step numbers from the notebooks;
- re-ran the EDA notebook and notebooks 03 and 05 (05 to update a plot label) in an environment built from `requirements.txt`, and confirmed that every printed result and results file was unchanged apart from floating-point noise below 10⁻¹⁴;
- updated the README, `results/NUMBERS_FOR_SLIDES.md`, `results/CHANGES.md` and this record.

## What the group did

Approve and verify each decision before letting Claude run it, discuss and evaluate the results before deciding if we want to keep or change our analysis method.

## Verification

We ran the notebooks manually, rechecked all numbers, and also evaluated the statistical reasoning for each code segment.

## Group members

Amanda
Declan
Felicia
Jeryl
Ryan
