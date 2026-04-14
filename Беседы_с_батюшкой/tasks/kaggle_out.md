# ── 3. Train OAT tokenizer (~2-3h, 300 epochs) ───────────────────────────────
# Запускаем из /kaggle/working — один venv, не создаём второй в oat/.venv
!uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment




    Error executing job with overrides: ['task/tokenizer=libero/libero10', 'training.num_epochs=300', 'logging.project=VLA-experiment']
Error in call to target 'oat.dataset.zarr_dataset.ZarrDataset':
TypeError('open() takes from 0 to 1 positional arguments but 2 were given')
full_key: task.tokenizer.dataset

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
