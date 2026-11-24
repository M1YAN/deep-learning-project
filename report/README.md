# English course report

[Read the PDF](paper.pdf). [paper.tex](paper.tex) is the editable manuscript and [references.bib](references.bib) contains its references. The report uses camera-ready formatting with page numbers from the [official CVPR 2024 v2 author kit](https://github.com/cvpr-org/author-kit/tree/CVPR2024-v2). The included `cvpr.sty` and `ieeenat_fullname.bst` are the upstream files from that tag.

The byline uses the repository author's handle. Edit the author block in `paper.tex` to update it.

## Rebuild the PDF

The [pipeline image](figures/pipeline-imagegen.png), vector plots, `ranking_rows.tex`, and `sweep_rows.tex` are included, so LaTeX compilation can run directly. From the repository root:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Tectonic can also compile the report with `tectonic paper.tex` from the report directory. To regenerate the experiment plots and both result tables from the experiment logs first, install Matplotlib and run `python report/build.py` from the repository root.

## Reproduce the experiment summaries

Set `GPUS` to the available GPU indices for parallel training. For data preparation and experiment settings, see the repository's [README.md](../README.md). Run from the repository root after installing the experiment dependencies in [requirements.txt](../requirements.txt).

```bash
bash scripts/run_experiments.sh course_256
bash scripts/run_experiments.sh course_256_long
python -m sidrec.evaluate course_256_long --device cuda --threads 2
bash scripts/run_neighbor_sweep.sh
python -m report.evaluate_activity --device cuda --threads 2
python report/build.py
```

The activity command replays saved best checkpoints and exports collided-target NDCG@10 by the number of pre-validation user interactions. The figure builder also accepts the original cohort summary already in the workspace.

The report labels `(λ, k) = (0.50, 3)` as the original reference setting, `(0.50, 1)` as the validation-selected positive-weight setting, and `(1.00, 3)` as the highest observed test mean. The implementation default follows the original reference setting.
