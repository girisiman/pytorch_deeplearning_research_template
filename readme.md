# PyTorch Workshop @MBUST

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

## 🚀 How to Run Training

After setting up the environment, run training using:

```bash
python scripts/train.py
```

## ⚙️ Configuration

All experiments are controlled via:

```bash
config/experiment.yaml
```

## 📓 Notebooks

Used for:
- Data exploration
- Visualization
- Error analysis
(Not for training code)

## 🔁 Reproducibility
 - Fixed seeds
 - YAML configs
 - Modular design

## 📦 Notes for Students
- Do NOT write all code in one file
- Do NOT hardcode hyperparameters & Always use configuration files
- Keep training code separate from model code
- Use notebooks only for analysis and visualization