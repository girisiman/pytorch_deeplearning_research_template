# PyTorch Research Template

This repository was created for a **PyTorch workshop for AI/DS Graduate Students at MBUST, Chitlang**.

It provides a clean, research-grade PyTorch project structure for deep learning experiments, reproducibility, and modular development.

---

## 🎯 Objective

The goal of this template is to teach:

- Research-grade PyTorch project structure
- Reproducible machine learning experiments
- Separation of concerns in ML codebases
- Configuration-driven development using YAML
- Proper use of scripts, notebooks, and source code
- Industry-style ML project organization

---

## 📁 Project Structure
config/ # Experiment configuration files (YAML)
src/ # Core source code
├── data/ # Dataset loading and preprocessing
├── models/ # Neural network architectures
├── training/ # Training and validation logic
└── utils/ # Helper functions (logging, config, metrics)

scripts/ # Entry point scripts (train, evaluate, predict)
notebooks/ # Data exploration and visualization
tests/ # Unit tests for code validation

data/ # Raw and processed datasets (ignored in git)
checkpoints/ # Saved models (ignored in git)
outputs/ # Results, plots, predictions
logs/ # Training logs

---

## 🚀 How to Run Training

After setting up the environment, run training using:

```bash
python scripts/train.py

⚙️ Configuration

All experiments are controlled via:

config/experiment.yaml
📓 Notebooks

Used for:

Data exploration
Visualization
Error analysis

(Not for training code)

🔁 Reproducibility
Fixed seeds
YAML configs
Modular design
👨‍🏫 Workshop

Created for AI/DS Graduate Students at MBUST, Chitlang.