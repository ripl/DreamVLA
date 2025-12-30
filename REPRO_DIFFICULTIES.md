## DreamVLA (CALVIN ABC→D) reproduction difficulties (succinct)

### Key blockers + what we did
- **Conda Terms-of-Service block**: `conda create` failed in the container due to ToS not accepted.
  - Fix: ran `conda tos accept --override-channels -c https://repo.anaconda.com/pkgs/main --system` and same for `.../pkgs/r` inside the persistent overlay.
- **GPU architecture mismatch (Blackwell sm_120)**: PyTorch `2.2.0+cu121` cannot run on sm_120 (“no kernel image…”).
  - Fix: ran eval on a non-Blackwell node (e.g., RTX A6000 sm_86) via Slurm constraint (`--constraint=48g`).
- **PyBullet EGL failure (headless GPU rendering)**: CALVIN env init failed with `failed to EGL with glad.`
  - Root cause: inside Apptainer, EGL couldn’t locate the NVIDIA ICD because the container did not have the GLVND vendor configs (`/usr/share/glvnd/egl_vendor.d`).
  - Fix (GPU render): bind the host vendor configs into the container (e.g. add `--bind /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d` to the `apptainer exec --nv ...` command) and set `use_egl: true` in `validation/.hydra/merged_config.yaml`.
  - Workaround (slow render): set `use_egl: false` so PyBullet uses the non-EGL renderer.
- **“No dataset” constraint vs DreamVLA eval code**: `eval_calvin.py` tried to build a training dataloader (`get_calvin_dataset`) which requires `.../training` data.
  - Fix: removed that `get_calvin_dataset(...)` call and used a stub dataset containing only `validation/.hydra/merged_config.yaml`.
- **Missing deps at runtime**: hit `ModuleNotFoundError: quaternion` from CALVIN.
  - Fix: installed `numpy-quaternion` (plus a few CALVIN runtime deps like `hydra-colorlog`, `numba`, `pandas`) into the overlay conda env.
- **Overlay locking**: `apptainer` can’t mount the same overlay RW concurrently.
  - Fix: run eval via Slurm; avoid interactive RW mounts while the job is running.
- **Progress visibility**: default eval didn’t stream per-episode stats.
  - Fix: added per-sequence CSV logging in `utils/eval_utils_calvin.py` (episode idx, success/failure, running SR, walltime).

### Artifacts / paths used
- **Base image**: `/share/data/ripl/projects/Policy_Eval_Done_Right/base.sif`
- **Persistent overlay**: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/dreamvla_overlay.img`
- **CALVIN source (mounted)**: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/calvin`
- **Stub dataset**: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/calvin_dataset_stub/task_ABC_D`
- **Weights cache**:
  - ViT MAE: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/checkpoints/vit_mae/mae_pretrain_vit_base.pth`
  - DreamVLA ABC→D: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/checkpoints/dreamvla_calvin_abc_d/dreamvla_dynamic_depth_semantic.pth`
- **Slurm eval script**: `/share/data/ripl/projects/Policy_Eval_Done_Right/cache/eval_dreamvla_abc_d.sbatch`

