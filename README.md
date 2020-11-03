# dnn-dot
Deep Neural Network for Diffuse Optical Tomography

# Framework
1. Python 3.5-3.6
2. Keras 2.2-2.3
3. Tensorflow 1.x should work

# Hyperparameters
1. Loss function: Weighted MSE
2. Optimizer: Adam (beta_1 = 0.5)
3. Learning rate: 0.0002
4. Batch size: 32
5. Epochs: 200

# Functional Block Diagram
This is an implementation of two-dimensional FD (Frequency Domain) DOT (Diffuse Optical Tomography) image reconstruction with measurement system of ring-scanning configuration using deep learning as an alternative to the conventional method (using FEM based on diffuse equation).
![alt text](https://raw.githubusercontent.com/diannatarahman/dnn-dot/master/images/block%20diagram.png)

# Learning curves
Training vs validation data (both using simulation data)
![alt text](https://raw.githubusercontent.com/diannatarahman/dnn-dot/master/images/learning%20curves.png)

# Results
Inclusion can be located for experimental data using only simulation data to train
![alt text](https://raw.githubusercontent.com/diannatarahman/dnn-dot/master/results/Figure_30.png)
![alt text](https://raw.githubusercontent.com/diannatarahman/dnn-dot/master/results/Figure_40.png)

# References
1. [DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction](https://ieeexplore.ieee.org/document/8233175/)
@article{yang2018_dagan,
	author = {Yang, Guang and Yu, Simiao and Dong, Hao and Slabaugh, Gregory G. and Dragotti, Pier Luigi and Ye, Xujiong and Liu, Fangde and Arridge, Simon R. and Keegan, Jennifer and Guo, Yike and Firmin, David N.},
	journal = {IEEE Trans. Med. Imaging},
	number = 6,
	pages = {1310--1321},
	title = {{DAGAN: deep de-aliasing generative adversarial networks for fast compressed sensing MRI reconstruction}},
	volume = 37,
	year = 2018
}
