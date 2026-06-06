# Deep Reinforcement Learning for Heterogeneous Electric-Diesel Fleet Routing

Replication and extension of **POMO** (Policy Optimization with Multiple Optima, Kwon et al., NeurIPS 2020) to the Heterogeneous Electric-Diesel Vehicle Routing Problem with Time Windows (HEVRPTW).

**Course:** ARI5002 Optimization for Artificial Intelligence  
**Author:** Egemen Eroglu  
**Deadline:** June 28, 2026

---

## Overview

This project implements POMO from scratch in PyTorch and extends it with:
- **VRPTW** — time-window feasibility masking (5-dim node features)
- **EV range constraint** — cumulative distance mask for electric vehicles
- **Heterogeneous fleet** — vehicle-type embeddings (SCV / MCV / LCV / LCV\_BEV)

Models are trained on synthetic instances and evaluated on real DHL Istanbul operational data (January 2026, 905 route instances, 3 depots).

### Key results

| Problem | n | Method | Mean Length | vs OR-Tools |
|---------|---|--------|------------|-------------|
| CVRP | 20 | POMO greedy | 4.469 | −7.81% |
| CVRP | 20 | POMO +8×aug | 4.327 | **−10.75%** |
| CVRP | 50 | POMO +8×aug | 8.922 | — |
| VRPTW | 20 | POMO +8×aug | 4.579 | **−9.14%** |
| VRPTW | 50 | POMO +8×aug | 9.390 | — |

EV range constraint at R=2.0 increases tour length by +8.8% (n=20). Real-world evaluation on DHL data reveals significant distribution shift; top-performing instances show 41–54% route reduction over driver baselines.

---

## Project Structure

```
.
├── configs/
│   └── default.yaml          # Model hyperparameters and training config
├── data/
│   ├── raw/                  # DHL_Araclar.xlsx, stop data .xlsm (not committed)
│   ├── processed/            # fleet_registry.pkl, real_instances.pkl, benchmark CSVs
│   ├── fleet_parser.py       # DHL vehicle registry → fleet dict
│   ├── stop_parser.py        # Stop data cleaning and time-window parsing
│   ├── instance_builder.py   # VRP instance construction (Haversine distance matrix)
│   └── baseline_extractor.py # DHL driver route reconstruction and metrics
├── pomo/
│   ├── env.py                # VRPEnv — multi-trip CVRP/VRPTW/EV state machine
│   ├── model.py              # AttentionEncoder + PomoDecoder
│   ├── trainer.py            # POMO REINFORCE training loop
│   └── inference.py          # Greedy rollout + 8× augmented decoding
├── eval/
│   ├── benchmark_eval.py     # Synthetic CVRP/VRPTW benchmark vs OR-Tools/NN
│   ├── distribution_analysis.py  # Training vs real-data distribution shift
│   └── merge_benchmark.py    # Merge aug/no-aug benchmark CSVs
├── viz/
│   ├── route_map.py          # Folium HTML maps (DHL vs POMO routes)
│   ├── gen_top_maps.py       # Generate comparison maps for top instances
│   ├── benchmark_plots.py    # Paper figures (bar charts, sensitivity plots)
│   ├── training_plots.py     # Loss curves and reward plots
│   ├── figures/              # Generated PNG figures (benchmark, EV, distribution)
│   └── maps/                 # Generated HTML route maps
├── notebooks/
│   └── colab_training.ipynb  # Full training + evaluation notebook for Google Colab
├── colab_results/            # Trained model checkpoints (not committed to git)
│   ├── best_cvrp.pt
│   ├── best_vrptw.pt
│   └── best_finetune.pt
├── documents/
│   └── POMO- Policy Optimization with Multiple Optima for Reinforcement Learning.pdf
├── report/
│   ├── main.tex              # IEEE-format course report (Overleaf)
│   └── references.bib
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. PyTorch is installed separately (see [pytorch.org](https://pytorch.org/) for your CUDA/MPS version):

```bash
pip install torch torchvision
```

---

## Usage

### Training (local)

```bash
# Train CVRP model (n=50, 100 epochs)
python train.py --config configs/default.yaml --n_nodes 50 --epochs 100

# Resume from checkpoint
python train.py --resume checkpoints/best_cvrp.pt
```

### Training (Google Colab — recommended)

Open `notebooks/colab_training.ipynb` in Google Colab. The notebook handles installation, training both CVRP and VRPTW models, running the full benchmark, and downloading results.

### Evaluation

```bash
# Synthetic benchmark (CVRP + VRPTW, with OR-Tools)
python eval/benchmark_eval.py \
    --cvrp_checkpoint  colab_results/best_cvrp.pt \
    --vrptw_checkpoint colab_results/best_vrptw.pt

# Skip OR-Tools (faster)
python eval/benchmark_eval.py --no_ortools

# Distribution shift analysis
python eval/distribution_analysis.py

# Generate benchmark figures
python viz/benchmark_plots.py
```

### Route Maps

```bash
# Generate POMO vs DHL comparison maps for top instances
python viz/gen_top_maps.py

# Custom route map
python viz/route_map.py --route CECD --depot CET --checkpoint colab_results/best_cvrp.pt
```

---

## Data

The raw DHL data (`data/raw/`) is proprietary and not included in this repository. To reproduce the real-instance results you need:

- `DHL_Araclar.xlsx` — vehicle fleet registry
- `Copy of PuD_STOP_DetailExport_cleaned.xlsm` — stop data (January 2026)

Place both files in `data/raw/` and run:

```bash
python data/fleet_parser.py
python data/stop_parser.py
python data/instance_builder.py
```

Synthetic benchmarks require no additional data.

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Encoder | 6-layer Transformer, d=128, 8 heads, d\_ff=512 |
| Decoder | Single-layer attention, tanh clipping (C=10) |
| Input dim | 3 (CVRP: x, y, d/C) or 5 (VRPTW: +e/T, l/T) |
| Training | POMO REINFORCE, shared within-instance baseline |
| Batch | 64 instances × n starts |
| Optimizer | Adam, lr=1e-4, 100 epochs |
| Inference | Greedy argmax + optional 8× coordinate augmentation |

---

## References

Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Min, S., & Gwon, Y. (2020). **POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.** *NeurIPS 2020.*

Kool, W., van Hoof, H., & Welling, M. (2019). **Attention, Learn to Solve Routing Problems!** *ICLR 2019.*
