# Mobula

[![](https://api.travis-ci.org/wkcn/mobula.svg?branch=master)](https://travis-ci.org/wkcn/mobula)
[![Coverage Status](https://coveralls.io/repos/github/wkcn/mobula/badge.svg?branch=master)](https://coveralls.io/github/wkcn/mobula?branch=master)

## What is it?
*Mobula* is a light deep learning framework on python.

It's **an efficent Python-DNN Implementation used numpy mainly**, and it's aimed to learn **how a neural network runs** :-)

## What can I do with it?
1. Deploy a Deep Neural Network, try to explore how it works when DNN is training or testing.
2. Implement some interesting layers quickly and make a test.

## Benefit

- Easy to Configure

    Mobula needs less dependence. It is implemented by numpy mainly, so you can setup it easily.

- Easy to modify

	Mobula is implemented by only Python. You can modify the code easily to implement what you want.

## How to install it?
```
pip install mobula
```

## Layers
*Mobula* has implemented these layers using numpy. It's efficient relatively on Python Implementation.

The Layers supports multi-input and multi-output.

#### Layers with Learning
- FC - Fully Connected Layer
- Conv - Convolution
- ConvT - Transposed Convolution
- BatchNorm
#### Layers without Learning
- Pool - Pooling
- Dropout
- Reshape
- Crop
#### Activation Layer
- Sigmoid
- ReLU
- PReLU
- SELU
- Tanh
- Softmax
#### Multi I/O Layer
- Concat
- Slice
- Eltwise
#### Cost Layer
- MSE - Mean Square Error
- CrossEntropy
- SigmoidCrossEntropy
- SoftmaxWithLoss 
- L1Loss
- SmoothL1Loss
- ContrastiveLoss
#### Evaluation Layer (No Backward)
- Accuracy (top_k)
#### Operators
- Add, Subtract
- Multiply
- Positive, Negative
- Exp, Log
- ReduceMean
- ReduceMax, ReduceMin

## Solvers

*Mobula* supports various solvers.

- SGD
- Momentum

## Quick Start

##### Notice: Recommend using Python in Anaconda, because of **Calculating Optimization numpy-mkl** in Anaconda.

The detail is in [Performance Analysis](docs/performance.md).

#### Digital Recognition
Let's construct a **Convolution Nerual Network** on *Mobula*! 

We use **LeNet-5** to solve *Digital Recognition* problem on Kaggle.

The score is above 0.99 in training for several minutes.

Firstly, you need to download the dataset train.csv and test.csv into **test/** folder. 

Secondly, constructing the **LeNet-5**.

The core code is that:

```python

import mobula
import mobula.layers as L
import mobula.solvers as S

data, label = L.Data([X, labels], "data", batch_size = 100)
conv1 = L.Conv(data, dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, pool = L.Pool.MAX, kernel = 2, stride = 2)
conv2 = L.Conv(pool1, dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool2, dim_out = 500)
relu3 = L.ReLU(fc3)
pred  = L.FC(relu3, "pred", dim_out = 10)
loss = L.SoftmaxWithLoss(pred, "loss", label = label)

# Net Instance
net = mobula.Net()

# Set Loss Layer
net.set_loss(loss)

# Set Solver
net.set_solver(S.Momentum())

# Learning Rate
net.lr = 0.2

```

The training and predicting codes are in **examples/** folders, namely **mnist_train.py** and **mnist_test.py**.

For training the network, 
```bash
python mnist_train.py
```

When the number of iterations is 2000, the accuracy on training set is above 0.99.

For predicting test.csv,  
```bash
python mnist_test.py
```

At Line 53 in *mnist_test.py*, `iter_num` is the iterations of the model which is used to predict test set. 

Enjoy it! :-)
