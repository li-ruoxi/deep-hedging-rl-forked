# Deep Hedging with Reinforcement Learning

This repository explores **reinforcement learning–based hedging strategies** for SPX & SPY options.  
We use Bloomberg/WRDS data (kept local), simulators, and RL agents to test risk management performance.  

---

## 📂 Project Structure

deep-hedging-rl/
│
├── data/ # (DO NOT push raw Bloomberg data, keep locally)
│ ├── raw/ # local-only (ignored in .gitignore)
│ └── processed/ # small sample/cleaned data for demos
│
├── notebooks/ # Jupyter/Colab notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_simulator_dev.ipynb
│ ├── 03_rl_training.ipynb
│ └── 04_results_analysis.ipynb
│
├── src/ # main Python modules
│ ├── data_pipeline/ # wrds/bloomberg loaders, cleaning functions
│ ├── simulator/ # hedging environment (Gym API)
│ ├── rl_agent/ # RL models, training loop
│ ├── evaluation/ # metrics, risk attribution, plots
│ └── utils/ # helpers, config, logging
│
├── experiments/ # configs & logs for each run
│ ├── configs/ # YAML/JSON config files
│ └── logs/ # training logs, wandb/mlflow outputs
│
├── reports/ # paper drafts, figures, slides
│
├── requirements.txt # pinned dependencies
├── environment.yml # optional, for conda setup
├── .gitignore # makes sure raw data is excluded
├── README.md # project overview, setup, usage
└── LICENSE # license if publishing



---

## ⚙️ Setup

### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate deep-hedging-rl

git lfs install && git lfs pull
pip install -r requirements.txt
python -m data_pipeline.cli all

Notes:
- Parquet support requires `pyarrow` (included in both `environment.yml` and `requirements.txt`).
- Large data files are tracked via Git LFS; ensure `git lfs pull` completes before running the pipeline.
- Used a 50/50 split for asset allocation in GAE and SPY. You can change allocation as you see fit in `notebooks/04_results.ipynb`.

### RL training workflow
1. Open `notebooks/03_rl_training.ipynb` and run the **parameter sweep** cell. It scans realistic trading knobs (`rebalance_every`, `slippage_bps`, `pos_limit`) and writes metrics to `models/sweep_run/<tag>/`.
2. Use the *final training* cell immediately below the sweep to retrain the best configuration with history logging and artifacts in `models/gae_<tag>/`. This is what `04_results.ipynb` expects.
3. Open `notebooks/04_results.ipynb` to reproduce evaluation plots, the long-SPY comparison, and the blended GAE + SPY analytics.
4. Adjust the `blend_weight` in the combined-strategy cell if you want to explore different GAE/SPY allocations; the shared `combo_summary` helper recomputes all risk metrics automatically.

### Testing
Run `pytest tests/test_env.py` to ensure the hedging environment still enforces `rebalance_every` and slippage cost semantics. Add new tests alongside that file when extending the simulator.
