# Deep learning-based Kalman Filter and linear quadratic Gaussian (LQG) optimal control of automated-guided vehicles (AGV)  

Automated-guided vehicles (AGV) is widely used for robotics and intelligent manufacturing. In indutrial applications, AGV can be simplified as a linear system. Linear quadratic Gaussian (LQG) optimal control consists of linear quadratic regulator (LQR) feedback control, and Kalman filter (KF) state observer for stochastic linear system.

Inspired by our previous work *Incorporating Transformer and LSTM to Kalman Filter with EM algorithm for state estimation* (see following paper and GitHub link), this repository combines deep learning models, Transformer and LSTM, for KF in LQG. EM-KF adopts expectation maximization (EM) algorithm for parameters estimation before Kalman filtering. This repository utilizes EM-KF to estimate the feedback matrix of LQR controller, instead of classical Ricatti equation-based LQR. 

This repository proposes kinematic simulations on path tracking in both cartesian and polar coordinate, and furthermore proposes simulation about mechanics. Please refer to the [notes](./notes.pdf) of this repository for implementation analysis and details.

Paper link: https://arxiv.org/abs/2105.00250

GitHub link: https://github.com/zshicode/Deep-Learning-Based-State-Estimation

## Usage

```
python main.py
```

## Requirements

The code has been tested running under Python3, with package PyTorch, NumPy, Matplotlib, PyKalman and their dependencies installed.

## Parameters

```python
# in main.py
epochs = 200
lr = 0.01
wd = 1e-5
random_state = np.random.RandomState(42)
torch.manual_seed(42)
T = 0.01
step = 200
model = 'T'
# T: Transformer, L: LSTM, TL: Transformer-LSTM, else: EM-KF
coord = 'C'
# C: cartesian (3-DOF), P: polar (2-DOF)
scene = 'K'
# K: kinematics, M: mechanics
vc = 1
xcoord0 = 1
ycoord0 = 1
theta0 = np.pi/12
```