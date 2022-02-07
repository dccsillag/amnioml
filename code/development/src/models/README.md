The models pipeline has the task of training and generating predictions on top of datasets produced by the data pipeline.

## Where models are implemented

A model `<MODEL_NAME>` is defined by simply creating a module
`src/models/<MODEL_NAME>/model.py`, containing a class called model `Model`,
which should be a child class of the `CommonLightningModule` class defined in
`model_utils`.

This `Model` class should define three important members:

- `NAME`: a well-formatted human-readable name for the model, which will be used in plots and tables.
- `__init__`: the constructor.
- `forward`: the model itself.

It's worth going a bit more in detail about `__init__`: Apart from
model-specific hyperparameters, it should receive `**kwargs`, and forward it
into the parent class constructor. Also, after setting all of the model
hyperparameters into as members, `self.save_hyperparameters()` should be
called. Here's an example:

```python
def __init__(self, model_specific_param_0, model_specific_param_1, **kwargs):
    super().__init__(**kwargs)

    self.model_specific_param_0 = model_specific_param_0
    self.model_specific_param_1 = model_specific_param_1

    self.save_hyperparameters()

    # ... any more model setup (e.g., constructing layers)
```

## Training Models

The `src/models/train.py` provides a common interface for training models. See
its module docstring for usage details.

It will create a directory for the trained/training model inside
`[personal/]models/<MODEL_NAME>/`, with a timestamp in the filename.

Among other things, this created directory contains a Tensorboard event file.
To launch Tensorboard to visualize training, you can use:

```sh
tensorboard --logdir=personal/models/<MODEL_NAME>
```

## Generating Model Predictions

Once you've trained a model, you can use `src/models/predict.py` to load its
checkpoint and save its predictions in a neat HDF file. Again, see the module
docstring at `src/models/predict.py` to see usage.

It will read the checkpoint from `[personal/]models/<MODEL_NAME>/<RUN_ID>/` and
write the output HDF at `[personal/]eval/<MODEL_NAME>/<RUN_ID>/`, which will
then be extensively used by the eval and confidence\_intervals pipeline.
