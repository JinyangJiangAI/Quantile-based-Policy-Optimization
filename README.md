# Quantile-based-Policy-Optimization
Official code for "Quantile-Based Deep Reinforcement Learning using Two-Timescale Policy Gradient Algorithms"

## Create Environment
The code can run on Windows and Linux with Python 3.9 and PyTorch 1.9.0+cu111.

## Experiment List
**fair_lottery**: comparison between QPO/QPPO and mean-based algorithms.

**beyond_greed**: comparison between QPO/QPPO and distributional reinforcement learning algorithms.

**inventory_management**: application in multi-echelon supply chain inventory management.

**toy_example**: extra experiment with all the theoretical assumptions satisfied.

**portfolio_management**: additional business application in finance.

## How to Run
For "toy_example", "fair_lottery", and "beyond_greed", you can start the training with
```
python train.py
```

For "inventory_management", you can start the training with
```
python train_ppo.py
python train_qppo.py
python train_qrdqn.py
```

For "portfolio_management", you can start the training with
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
