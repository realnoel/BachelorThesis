# Diffusion-Based Refinement for Robust Long-Horizon Predictions with Neural Operators

Code for my Bachelor's thesis at ETH Zurich (Computational Mechanics Group, Prof. Dr. Laura De Lorenzis; supervisor: Lukas Brock), December 2025.

The thesis studies whether **diffusion-based refinement** ([PDE-Refiner](https://arxiv.org/abs/2308.05732), Lippe et al., NeurIPS 2023) improves **long-horizon autoregressive predictions** of temperature fields in laser cutting. Unlike the closed systems in the original paper, this problem is an **open system**: the temperature evolution is driven by time-varying external forcing (laser power and tool motion).

**Main finding:** Gaussian input noise during training reduced the refiner's rollout error by 41% (rel. L1) and made training far less sensitive to hyperparameters. However, in this low-data setting (7 training trajectories), a plain FNO baseline remained more accurate at ~120× lower inference cost.

---

## Results

Autoregressive rollout of 165 time steps on the held-out test trajectory, errors in physical space:

| Model | Rel. L1 [%] | Rel. L2 [%] | Parameters | Inference / frame |
|---|---|---|---|---|
| FNO (Strategy 1, N = 3) | **2.83** | **6.45** | 0.9 M | 0.9 ms |
| CNO (Strategy 1, N = 3) | 5.25 | 8.16 | 8.0 M | 5.0 ms |
| Refiner, no input noise | 13.69 | 25.17 | 41.9 M | 110 ms |
| Refiner, Gaussian input noise | 8.10 | 13.01 | 41.9 M | 110 ms |

Timings measured on a single NVIDIA RTX 4090; refiner with K = 7 refinement steps.

Key observations:

- **Input noise stabilizes the refiner.** Without it, most hyperparameter settings diverge during rollout; with it, a much broader range stays stable.
- **More refinement steps hurt.** Increasing K degraded accuracy here, in contrast to the saturation trend reported by Lippe et al.
- **Likely reasons the refiner trails the FNO:** far less training data than the original benchmarks (~1.9k correlated samples from 7 trajectories vs. 2,048 trajectories), external forcing instead of closed dynamics, and smooth parabolic dynamics on which the FNO is already strong.

---

## Method

**Problem.** Predict the 2D temperature field T on a 44 × 44 grid in a frame co-moving with the laser. Each time step provides four channels: temperature T, laser power Q, and the frame shifts dx, dy. Temperature and power are log-transformed (`ln T`, `ln(Q + 1)`) and min-max normalized.

**Baselines.** FNO and CNO trained with a one-step MSE loss and rolled out autoregressively.

**Diffusion refiner.** The refiner predicts the residual between consecutive temperature fields as a DDPM with:

- **v-prediction** objective and an exponential noise schedule σ_k = σ_min^(k/K), implemented with the `diffusers` `DDPMScheduler`
- a **time-embedded CNO backbone** (FiLM conditioning on the refinement step k)
- **open-system conditioning**: the forcing fields (Q, dx, dy) are concatenated as spatial input channels, instead of the scalar parameter embeddings used in the original paper
- an **EMA** of the weights (decay 0.995) for validation

**Input perturbation variant.** During training only, Gaussian noise is added to the input temperature, T'_{t−1} = T_{t−1} + ε with ε ~ N(0, σ_T²). This exposes the model to imperfect inputs, similar to what it sees from its own predictions during rollout.

---

## Repository structure

```
BA_2D_Laser/
├── baseline_training.py                     # Train FNO / CNO baselines
├── refiner_without_pertubation_training.py  # Train the diffusion refiner
├── refiner_with_pertubation_training.py     # Train the refiner with Gaussian input noise
├── rollout.py                               # Autoregressive rollout + metrics on the test trajectory
├── configs/
│   └── default.yaml                         # All hyperparameters (read by every script)
├── dataloader/
│   └── dataloader_{1,2,3,4}.py              # Input strategies 1–4 (see below)
├── model/
│   ├── model_fno.py                         # 2D Fourier Neural Operator
│   ├── model_cno.py                         # 2D Convolutional Neural Operator
│   ├── model_cno_timeModule.py              # CNO with FiLM time embedding (refiner backbone)
│   └── model_cno_timeModule_utils.py        # Alias-free filtered LReLU activations
├── utils/
│   ├── utils_refiner.py                     # Refiner loss, EMA, metrics
│   ├── utils_train.py                       # Metrics and checkpoint helpers
│   ├── utils_images.py                      # Field and error plots
│   ├── utils_video.py                       # Rollout videos
│   └── torch_utils/, dnnlib/                # NVIDIA StyleGAN3 ops (filtered LReLU CUDA kernels)
├── requirements.txt
└── setup_euler.sh                           # Environment setup on the ETH Euler cluster
```

---

## Installation

Tested with Python 3.11 and CUDA 12.1.

```bash
git clone https://github.com/realnoel/BA_2D_Laser.git
cd BA_2D_Laser
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On the **ETH Euler cluster**, run `bash setup_euler.sh`. It loads `stack/2024-06 python_cuda/3.11.6` and creates a virtual environment in `~/venvs/ddpm_venv`.

**Note on the CNO activation.** With `activation: "cno_lrelu"`, the alias-free activation uses CUDA kernels from StyleGAN3 that are compiled on first use on the GPU. This requires a CUDA toolkit (`nvcc`) and `ninja` (`pip install ninja`).

---

## Data

The simulation data is **not included** in this repository. Please contact the author for access.

The scripts expect two HDF5 files in `data/` (names set in `configs/default.yaml`):

```
data/
├── pattern_paths_training_44_44.h5   # 7 trajectories, 1,934 time steps
└── pattern_paths_test_44_44.h5       # 1 trajectory, 170 time steps
```

Each file contains:

- normalization constants `min_t`, `max_t`, `min_q`, `max_q`, `min_shift`, `max_shift`
- groups `trajectory_*`, each with time steps `sample_*` containing:
  - `output`: temperature field (log scale)
  - `input_p`: laser power field
  - `dx`: frame shift in x and y (2 channels)

All fields are given on the same 44 × 44 grid. Normalization uses the constants from the training file.

---

## Input strategies

The dataloaders differ in which time steps of the forcing (Q, dx, dy) and of the temperature T the model receives. N sets the history length (and, for strategies 1 and 2, the number of predicted steps).

| Strategy | Forcing input | Temperature input | Target |
|---|---|---|---|
| 1 | past N and next N steps | past N steps | next N steps |
| 2 | next N steps | past N steps | next N steps |
| 3 | past N steps and next step | past N steps | next step |
| 4 | next step only | past N steps | next step |

The thesis uses **Strategy 1, N = 3** for the best baseline and **Strategy 3, N = 1** for the refiner.

---

## Usage

All scripts read `configs/default.yaml` and must be run from the repository root. Edit the config before each run.

### Train a baseline

Set in `configs/default.yaml`:

```yaml
training:
  model: "fno"        # or "cno"
  strategy: 1
  N: 3
```

Then run:

```bash
python baseline_training.py
```

### Train the diffusion refiner

Set in `configs/default.yaml`:

```yaml
training:
  model: "cno_temp"
  strategy: 3
  N: 1
refiner:
  k_max: 7                 # number of refinement steps K
  min_noise_std: 1e-3      # σ_min of the noise schedule
  state_noise_sigma: 1e-2  # σ_T of the Gaussian input noise (perturbation variant only)
```

Then run one of:

```bash
python refiner_without_pertubation_training.py   # refiner without input noise
python refiner_with_pertubation_training.py      # refiner with Gaussian input noise
```

The values above are the best configuration of the perturbation variant in the thesis. The best configuration without input noise used `min_noise_std: 1e-5` and `k_max: 7`. The remaining hyperparameters are listed in Appendix B of the thesis.

Each run creates `checkpoints/<YYYYMMDD_HHMMSS>/` with the model checkpoints, a CSV log of the training and validation metrics, and `logs/version_0/hparams.yaml`.

### Roll out a trained refiner

```bash
python rollout.py --ckpt <YYYYMMDD_HHMMSS> --strategy 3 --steps 165 --idx 0 --mode phys
```

| Argument | Description |
|---|---|
| `--ckpt` | Run folder name inside `checkpoints/` |
| `--strategy` | Input strategy the model was trained with (1–4) |
| `--steps` | Number of autoregressive rollout steps (165 in the thesis) |
| `--idx` | Start index in the test trajectory |
| `--mode` | `phys` for physical temperatures, `norm` for normalized values |
| `--device` | `auto`, `cpu` or `cuda` |

The script loads the most recently written checkpoint of the run. Results are saved to `results_val/<timestamp>_<ckpt>_<mode>_<steps>_<strategy>/`:

- `metrics.csv`: rel. L1, rel. L2 and MSE per time step, with the rollout average in the first row
- `rollout_times.csv`: inference time per step
- `rel_l1.png`, `rel_l2.png`, `mse.png`: error over the rollout
- `results_pred/`, `results_gt/`, `error_map/`: predicted, ground-truth and absolute-error fields per step

---

## Acknowledgements and third-party code

- **FNO**: adapted from the CAMLab ETH Zurich tutorial [*Operator Learning – Fourier Neural Operator*](https://github.com/camlab-ethz/AI_Science_Engineering).
- **CNO**: based on [camlab-ethz/ConvolutionalNeuralOperator](https://github.com/camlab-ethz/ConvolutionalNeuralOperator) (`CNO2d_simplified`, `CNO2d_temporal`).
- **Time embedding**: FiLM conditioning following [Poseidon](https://arxiv.org/abs/2405.19101) (Herde et al., 2024).
- **`utils/torch_utils/`, `utils/dnnlib/`**: from [NVIDIA StyleGAN3](https://github.com/NVlabs/stylegan3), subject to the NVIDIA Source Code License.
- **Diffusion scheduler**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers).

Training and evaluation were run on the ETH Zurich Euler cluster.

