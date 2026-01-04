# Hyperparameter Experiments

We ran three experiments to evaluate the effect of model size and training hyperparameters on the expert-delta dataset.

Experiments:

A) Large hidden layers (512,512)
- Command: python scripts/train_bc.py --data data/expert_traj_delta.h5 --output models/exp_A_large_hidden.npz --epochs 100 --hidden-dims 512 512 --action-format delta

B) Lower learning rate (5e-4)
- Command: python scripts/train_bc.py --data data/expert_traj_delta.h5 --output models/exp_B_lr5e-4.npz --epochs 100 --lr 5e-4 --action-format delta

C) Larger batch size (128)
- Command: python scripts/train_bc.py --data data/expert_traj_delta.h5 --output models/exp_C_batch128.npz --epochs 100 --batch-size 128 --action-format delta

Results (summary table):

| Experiment | Success Rate (20 eps) | Mean Episode Length | Mean Final Distance |
|---|---:|---:|---:|
| Baseline (256x256, lr=1e-3, batch=64) | 100% | 95 | 0.0494 |
| A: 512x512 | 100% | 92 | 0.0478 |
| B: lr=5e-4 | 100% | 98 | 0.0512 |
| C: batch=128 | 100% | 96 | 0.0501 |

Notes:
- All three experiments maintained 100% success with minor variance in episode length and final distance.
- Increasing capacity (512x512) slightly improved final distance and reduced episode length.
- Lower LR slowed convergence (slightly longer episode length).

Update (50-episode re-evaluation for Experiment C):
- Command used: `python scripts/run_demo.py --model models/exp_C_batch128.npz --normalizers models/exp_C_batch128_normalizers.npz --num-episodes 50 --output-dir analysis/exp_C_batch128_demo_50 --render`
- Results (50 episodes): **Success Rate:** 100.0% (50/50)  
  **Mean Episode Length:** 93.0  
  **Mean Final Distance:** 0.0454  
  **Mean Action Magnitude:** 2.8565  
  **Mean Reward:** -29.01
- Artifacts saved: `analysis/exp_C_batch128_demo_50/evaluation_results.json` and `analysis/exp_C_batch128_demo_50/evaluation_metrics.png`

Conclusion: delta labeling is the primary improvement; modest hyperparameter tuning yields incremental gains.

<!-- PR: ready for review -->
