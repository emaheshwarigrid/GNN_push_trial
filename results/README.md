# Results Tables

This folder stores CSV exports from the architecture sweeps.

## Layout

| Folder | Meaning |
| --- | --- |
| `60_20_20_split/` | Exploratory sweep results using the broader validation/test split |
| `80_10_10_split/` | Final-regime sweep results using more training data |

## How To Use These Files

Each CSV is the compact summary of a notebook sweep. They are useful for:

- checking the best recorded metrics per architecture,
- understanding which hyperparameter regions were strong,
- comparing how the same architecture behaves under different split choices.

## Why Keep CSVs Separate From Notebooks

The notebooks preserve the narrative and plots. The CSVs preserve the searchable evidence.

That split makes the repository easier to review because a grader or teammate can inspect:

- the full training story in the notebooks,
- the sortable results in tabular form.
