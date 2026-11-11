# Master-Sim

**Master-Sim** is a research MVP for learning precision-assembly robot policies in simulation. The long-term goal is Sim-to-Real peg-in-hole assembly; the current baseline is a MuJoCo Panda reach-task trained with Behavior Cloning.

## Overview
This project aims to bridge the gap between simulation and reality (Sim-to-Real) for tasks requiring delicate force control, such as peg-in-hole assembly. It does not yet implement a full peg, hole, or gripper task; see the resume plan for the current scope and next milestones.

## Getting Started

### Prerequisites
- Python 3.11+
- MuJoCo

### Installation

Create an isolated Python 3.11 environment, then install the pinned dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify the baseline in headless mode:

```bash
.venv/bin/python -m pytest
```

Rendering tests require an available OpenGL display context and are skipped in headless macOS sessions.

## Roadmap
Start with [docs/RESUME_PLAN.md](docs/RESUME_PLAN.md) for the verified current status and prioritized v0.1.0 plan. The original long-term roadmap remains in [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md).

## Development workflow

This is currently a main-first, single-developer project. See [docs/GIT_STRATEGY.md](docs/GIT_STRATEGY.md) for the commit, verification, and future-release branching policy.

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

