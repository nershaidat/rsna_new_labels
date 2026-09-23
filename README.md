# Knee MRI: hierarchical MIL with 12 weak labels

Predict study-level abnormalities by pooling slices within series and series
within studies. Labels are attached to studies, not copied to individual slices.

## Files

| File | Purpose |
| --- | --- |
| [knee_mri_mil12_v2.ipynb](knee_mri_mil12_v2.ipynb) | Full trainer in ordered Jupyter cells, with editable settings and separate audit/training controls |
| [knee_mri_mil12_v2.py](knee_mri_mil12_v2.py) | Original command-line trainer, unchanged |
| [WeakLabels_v3_4_training_targets_format.csv](WeakLabels_v3_4_training_targets_format.csv) | Included study labels, unchanged |
| [docs/archive/knee_mri_mil12_v1_README.txt](docs/archive/knee_mri_mil12_v1_README.txt) | Historical v1 instructions; use the v2 instructions here |
| [docs/notebook_validation.md](docs/notebook_validation.md) | Conversion checks and limits of verification |

The notebook stays beside the script and label CSV to keep relative paths simple.
The census SQLite checkpoint and DICOM images must be supplied separately.

## Jupyter workflow

1. Open Jupyter from this repository directory and select your MRI Python
   environment (Python 3.10+).
2. Open `knee_mri_mil12_v2.ipynb` and run the definition cells in order.
3. Edit **Notebook settings**, especially `CENSUS_PATH`, `LABELS_PATH`, and
   `RUN_DIR`, then execute that cell. Use paths accessible from the kernel:
   `/mnt/y/...` for WSL, or a Windows path when using a Windows kernel.
4. Set `RUN_AUDIT=True` in **Run audit**, execute it, and inspect the output CSVs.
5. Set `RUN_TRAINING=True` in **Run training** and execute it when ready.

Both controls initially default to `False`. An initial **Run All** loads the
definitions and prepares arguments without starting an audit or training.
Once enabled, subsequent **Run All** executions will perform those actions.
Training starts fresh each time; it does not resume automatically. Use a new
run directory for a different experiment to avoid overwriting prior results.

Audit needs `numpy` and `pandas`. Training additionally needs compatible `torch`,
`torchvision`, and `pydicom`; compressed DICOMs may need decoder plugins. Excel
input needs an appropriate pandas Excel engine (`openpyxl` for `.xlsx`).
Use your existing CUDA-capable environment for GPU training. Pretrained weights
may download on the first run. Keep `NUM_WORKERS=0` for the notebook, especially
under Windows/WSL; the trainer's nested classes and collate lambda are not
portable to spawn-based multiprocessing.

The notebook includes the actual implementation, not just a command that runs
the script. Treat the script as the source of truth for future code changes;
changes to one file do not automatically update the other.

## Command-line workflow

From the repository directory, first run an audit (replace the census path):

```bash
python knee_mri_mil12_v2.py audit \
  --census "/path/to/knee_mri_census.xlsx.checkpoint.sqlite" \
  --labels "WeakLabels_v3_4_training_targets_format.csv" \
  --run-dir "results/mil12_v2" \
  --split-mode random
```

Then train using the same input paths and split settings:

```bash
python knee_mri_mil12_v2.py train \
  --census "/path/to/knee_mri_census.xlsx.checkpoint.sqlite" \
  --labels "WeakLabels_v3_4_training_targets_format.csv" \
  --run-dir "results/mil12_v2" \
  --split-mode random \
  --epochs 20 \
  --max-series 8 \
  --slices-per-series 12 \
  --num-workers 0 \
  --precision bf16
```

`random` preserves the original CLI default and splits by study. Choose
`scanner` for normalized scanner-domain holdout, or `source` for source-group
holdout; these require at least three distinct groups. Add `--freeze-encoder`
for a frozen-encoder experiment. See `python knee_mri_mil12_v2.py train --help`
for the full list of options.

## Labels and outputs

Targets: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, and Fracture.

The loader accepts `StudyInstanceUID` plus either `<Target>_P` or `<Target>`
columns, preferring `<Target>_P` when both exist. Extra `__confidence`, `__state`,
and `__graded_*` columns are ignored. The existing trainer derives its weights
from the target probabilities:

\[
w=2\lvert p-0.5\rvert.
\]

Thus `p=0.5` carries zero training weight. This conversion preserves that rule.

Audit writes `split.csv`, `series_manifest.csv`, `study_manifest.csv`,
`label_summary.csv`, `scanner_domain_split.csv`, and `split_label_summary.csv`.
Training writes checkpoints (`best.pt`, `last.pt`), `config.json`, `history.csv`,
validation metrics, test metrics when a test split exists, and
`val_attention_top_slices.csv`. Attention is an audit clue, not ground-truth
lesion localization.
