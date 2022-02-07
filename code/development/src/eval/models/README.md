This part of the eval pipeline evaluates our models.

Each file in here is a script, and corresponds to some unique
experiment/analysis. See their module docstrings for an explanation of what
each one is.

There is one general rule for model evaluation scripts, concerning where plots/tables are placed:

- If the analysis is specific to a model, then we'll place the output into that model's output directory:
  - If the output is a plot, it should be stored in `[personal/]eval/<MODEL_NAME>/<RUN_ID>/viz/<EXPERIMENT_NAME>.png`;
  - If the output is a table, it should be printed and then stored in `[personal/]eval/<MODEL_NAME>/<RUN_ID>/tables/<EXPERIMENT_NAME>.tex`.
- If the analysis is *not* specific to a model (e.g., it compares different models), then we'll place the output into the global evaluation directory:
  - If the output is a plot, it should be stored in `[personal/]eval/viz/<EXPERIMENT_NAME>.png`;
  - If the output is a table, it should be printed and then stored in `[personal/]eval/tables/<EXPERIMENT_NAME>.tex`.
