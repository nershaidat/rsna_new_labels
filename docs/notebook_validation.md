# Notebook conversion validation

Source revision: `5f701ec3200bc0ecd5159c7d6eb34f5e0b59e5fc`.

## Scope

The repository contained one Python file, `knee_mri_mil12_v2.py`. Its notebook
counterpart contains 35 cells (17 code, 18 Markdown). Original code cells are
tagged `original-source`; notebook settings and action cells have separate tags.
The Python script, label CSV, and license remain unchanged. The historical
v1 README was moved without content changes to `docs/archive/`.

The only omitted source statement is the final
`if __name__ == "__main__": main()` guard. The `main` function itself remains
defined. Notebook controls call the original parser with explicit arguments,
then invoke the original audit or training function. This avoids parsing
Jupyter's kernel arguments and prevents automatic training on initial Run All.
The original module docstring remains verbatim, including its historical example
paths; use the new README and Notebook settings for the notebook workflow.

## Checks completed

- Validated the notebook with `nbformat.validate` (format 4.5).
- Compiled every code cell independently.
- Concatenated `original-source` cells and checked exact text equality with the
  script before its final entry-point guard. Compared Python abstract syntax
  trees as an additional structural check.
- Verified the source SHA-256 recorded in notebook metadata.
- Executed all default cells in order using IPython, checking that neither
  audit nor training started and no results directory was created.
- Loaded the actual repository CSV: 4,407 studies with the expected 12 targets.
- Checked information weights for probabilities 0, 0.2, 0.5, 0.8, and 1.
- Created a temporary synthetic census with 24 studies, two slices each, and
  six scanner/source groups, linked to actual CSV study identifiers. Synthetic
  image paths intentionally did not point to DICOM files; audit does not decode
  images.
- Ran notebook and original CLI audits for `random`, `scanner`, and `source`.
  All six output CSVs were byte-for-byte identical for each mode (18 comparisons).
- Verified study uniqueness across splits and whole-group holdout for the
  scanner and source fixtures.
- Verified that audit execution did not import PyTorch.
- Kept delivered cells unexecuted with empty outputs and null execution counts.

## Limits

The environment rejected both TCP and IPC socket creation for a standalone
Jupyter kernel. Cell execution therefore used IPython in-process; it was not a
browser-based Jupyter session. Full training, GPU operation, DICOM decoding,
pretrained-weight loading, and checkpoint reloads were not tested. The user's
real census database and images were unavailable. The synthetic audit validates
conversion equivalence, not the scientific quality of the data split or model.

Future changes to the script must be synchronized with the notebook; these are
two separate files. The `original-source` tags and source hash support checking
that correspondence.
