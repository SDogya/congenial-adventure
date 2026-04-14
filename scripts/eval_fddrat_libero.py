"""Evaluate a trained FD-DRAT checkpoint on LIBERO10.

Usage:
    python scripts/eval_fddrat_libero.py -c model.ckpt -o eval_out/
    python scripts/eval_fddrat_libero.py -c checkpoints/ -o eval_out/ -n 3
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, 'oat'))
    sys.path.insert(0, os.path.join(ROOT_DIR, 'hnet'))
    os.chdir(ROOT_DIR)

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import time
import json
import pathlib
import click
import torch
import wandb
import numpy as np
from typing import List, Optional

from oat.env_runner.libero_runner import LiberoRunner


@click.command()
@click.option('-c', '--checkpoint', required=True,
              help=".ckpt file or directory with .ckpt files")
@click.option('-o', '--output_dir', required=True,
              help="output directory for eval_log.json and videos")
@click.option('-n', '--num_exp', default=1,
              help="number of rollout repetitions to average over")
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=50,
              help="total rollout episodes (≥ #tasks for full LIBERO10 coverage)")
@click.option('--n_test_vis', default=5,
              help="how many episodes to record as video")
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int,
    device: str,
    n_test: int,
    n_test_vis: int,
):
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect checkpoint paths
    if os.path.isdir(checkpoint):
        ckpts: List[str] = sorted([
            os.path.join(checkpoint, f)
            for f in os.listdir(checkpoint)
            if f.endswith('.ckpt') and f != 'last.ckpt'
        ])
    else:
        ckpts = [checkpoint]

    base_output_dir = output_dir
    for ckpt in ckpts:
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir

        # Load policy from Lightning checkpoint
        from src.core.system import LitSystem
        system = LitSystem.load_from_checkpoint(ckpt, map_location=device, weights_only=False)
        policy = system.model
        policy.to(torch.device(device))
        policy.eval()

        # Build LIBERO10 runner
        # n_parallel_envs caps simultaneous MuJoCo workers to avoid OOM.
        # Runner runs in batches of n_parallel_envs, so n_test is still fully covered.
        env_runner = LiberoRunner(
            output_dir=output_dir,
            task_name="libero10",
            n_test=n_test,
            n_test_vis=n_test_vis,
            n_obs_steps=policy.n_obs_steps,
            n_action_steps=policy.n_action_steps,
            n_parallel_envs=4,
        )

        # Latency profiling hook
        latency_ms: List[float] = []
        _orig = policy.predict_action

        def _timed(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _orig(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - t0) * 1000.0)
            return out

        policy.predict_action = _timed

        # Run rollouts
        all_runs = []
        runner_log = env_runner.run(policy)
        all_runs.append({k: v for k, v in runner_log.items()
                         if not isinstance(v, wandb.sdk.data_types.video.Video)})
        print(f"Exp 1 — mean_success_rate = {runner_log['mean_success_rate']:.3f}")

        for i in range(num_exp - 1):
            log = env_runner.run(policy)
            print(f"Exp {i+2} — mean_success_rate = {log['mean_success_rate']:.3f}")
            all_runs.append({k: v for k, v in log.items()
                             if not isinstance(v, wandb.sdk.data_types.video.Video)})
            for k, v in log.items():
                if isinstance(v, wandb.sdk.data_types.video.Video):
                    runner_log.setdefault(k, [])
                    if isinstance(runner_log[k], list):
                        runner_log[k].append(v)
                else:
                    runner_log[k] = runner_log.get(k, 0) + v

        env_runner.close()

        # Aggregate statistics
        numeric_keys = list(all_runs[0].keys())
        json_log: dict = {"checkpoint": ckpt, "num_exp": num_exp}

        for k in numeric_keys:
            vals = [r[k] for r in all_runs]
            json_log[f"{k}_mean"] = float(np.mean(vals))
            if num_exp > 1:
                json_log[f"{k}_std"] = float(np.std(vals, ddof=1))
                json_log[f"{k}_stderr"] = float(np.std(vals, ddof=1) / np.sqrt(num_exp))

        if latency_ms:
            json_log["latency_p99_ms"] = float(np.percentile(latency_ms, 99))
            json_log["latency_mean_ms"] = float(np.mean(latency_ms))

        # Video paths
        for k, v in runner_log.items():
            if isinstance(v, list):
                for i, vid in enumerate(v):
                    if isinstance(vid, wandb.sdk.data_types.video.Video):
                        json_log[f"{k}_{i}"] = vid._path

        out_path = os.path.join(output_dir, "eval_log.json")
        json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    eval_policy_sim()
