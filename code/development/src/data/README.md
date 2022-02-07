The data pipeline has two fronts:

- Read raw data (located in `data/raw/`) and process it into a neat dataset (located in `data/processed/<dataset_name>`);
- Utilities that work exclusively with the data.

## Data Processing

There are two scripts for this:

- `src/data/create_dataset-segmentation.py`
- `src/data/split_dataset.py`

See their module docstrings for information on how to use them.

These are the commands you would run to, for example, generate the dataset `segmentation_700`, which contains 700 subjects:

```sh
python src/data/create_dataset-segmentation.py -n 700
python src/data/split_dataset.py personal/processed/segmentation_700
```

## Data-related Utilities

Right now, this is just the `src/data/visualize_dataset-segmentation.py` script. See its module docstring for information on it.
