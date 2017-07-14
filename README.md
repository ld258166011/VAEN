# Variational Autoencoder Network
Variational Autoencoder Network (**VAEN**) implementation on IMTD17 datasets to realize traffic identification.

## Introduction
3 files are available in this repository: 

| File Name  | Description                     | Size (bytes) |
| :--------- | :------------------------------ | :----------- |
| model.py   | Variational Autoencoder Network | 8,259        |
| figure.py  | Figure drawing helper functions | 7,351        |
| main.py    | Main process in TensorFlow      | 10,676       |

## Application

### Sample Reconstruction
Sample reconstruction of (a)Alipay, (b)Kugou, (c)Weibo traffic image.

![image](figures/1.png)

### Latent Representation
2-D Latent representation distribution.

![image](figures/2.png)

Sample reconstruction from 2-D latent representation.

### Traffic Identification
Classification result under different threshold.

Classification Result of App Traffic.
