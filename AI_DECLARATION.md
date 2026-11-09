# Declaration of AI tool use

> **DRAFT. To be completed by the group.** Replace every `[TODO: ...]` with your own words. Do not submit statements that do not reflect what you actually did.

## Tools used

- Claude / Claude Code (Anthropic), used for code assistance and review.
- [TODO: list any other AI tools used, or state "none".]

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

Claude Code also drafted the markdown explanations in the notebooks and this README. An attempt to check the wording of Cortez et al. (2009) online failed, so no claim about that paper's cut-off was made.

## What the group did

[TODO: describe the decisions and work done by group members, for example the research question, choice of application and cost ratio, model selection rule, interpretation of results and the written report.]

## Verification

[TODO: describe how the group checked AI-assisted code and text, for example re-running notebooks, checking numbers against outputs, and reviewing the statistical reasoning.]

## Group members

[TODO: names and student IDs]
