# Quantile-based-Policy-Optimization
Official codes for "Quantile-Based Deep Reinforcement Learning using Two-Timescale Policy Gradient Algorithms"

## Create Environment
The codes are implementable on both Windows and Linux with Python 3.9 and PyTorch 1.9.0+cu111.

## How to Run
For the first experiment "zero_mean", you can start the training with
```
python train.py
```

For the last two experiments, "portfolio_management" and "inventory management", you can start the training with
```
python train_ppo.py
python train_qppo.py
```

## BibTeX

```bibtex
@article{jiang2023quantile,
  title={Quantile-Based Deep Reinforcement Learning using Two-Timescale Policy Gradient Algorithms},
  author={Jiang, Jinyang and Hu, Jiaqiao and Peng, Yijie},
  journal={arXiv preprint arXiv:2305.07248},
  year={2023}
}
```
