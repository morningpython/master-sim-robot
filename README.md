# Master-Sim

**Master-Sim** is a project dedicated to developing high-fidelity synthetic data and pre-trained policies for precision assembly tasks in industrial robotics.

## Overview
This project aims to bridge the gap between simulation and reality (Sim-to-Real) for tasks requiring delicate force control, such as peg-in-hole assembly.

## Getting Started

### Prerequisites
- Python 3.8+
- MuJoCo

### Installation
```bash
pip install -r requirements.txt
```

## Roadmap
See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the detailed business and technical roadmap.

## Quick Start (recommended)

Train (recommended: action labels as deltas):

```bash
python scripts/generate_expert_data.py --num-trajectories 200 --steps 200 --output data/expert_traj_delta.h5 --action-format delta
python scripts/train_bc.py --data data/expert_traj_delta.h5 --output models/expert_trained_delta.npz --epochs 100 --action-format delta
```

Demo:

```bash
python scripts/run_demo.py --model models/expert_trained_delta.npz --normalizers models/expert_trained_delta_normalizers.npz --num-episodes 20
```

Notes:
- We recommend using `action_format=delta` (predicting joint deltas) for Behavior Cloning with IK-derived expert data: it produced stable policies with 100% success in our experiments (see `docs/experiments/delta_vs_qpos_report.md`).
- To retain backward compatibility you can still generate or train using `--action-format qpos`.

