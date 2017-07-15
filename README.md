# Mobula

## What is it?
Mobula is a light deep learning framework on python. It is aimed to learn **how a neural network runs**, and implement ideas quickly. 

## What can I do with it?
1. Digital Recognition

## Layers
Mobula has implemented these layers using numpy. It's efficient relatively on Python Implementation.
#### Layers with Learning
- FC
- Convolution
#### Layers without Learning
- Pooling
- Dropout
#### Activation Layer
- Sigmoid
- ReLU
- PReLU
- Tanh
- Softmax
#### Cost Layer
- Mean Square Error
- CrossEntropy
- SigmoidCrossEntropy
- SoftmaxWithLoss 

## Benefit

- Easy to Configure
	Mobula needs less dependence. It is implemented by numpy mainly, so you can setup it easily.

## Quick Start
#### Digital Recognition
Let's construct a Convolution Nerual Network on Mobula! 

We use LeNet-5 to solve *Digital Recognition* problem on Kaggle.

The score is above 0.99 in training for several minutes.

Firstly, you need to download the dataset train.csv and test.csv. 
Secondly, constructing the LeNet-5.
The core code is that:

```python
from mobula import Net
import mobula.layers as L

data = L.Data(X, "Data", batch_size = 100, label = labels)

conv1 = L.Conv(data, "Conv1", dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
conv2 = L.Conv(pool1, "Conv2", dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool2, "fc3", dim_out = 500)
relu3 = L.ReLU(fc3, "relu3")
pred  = L.FC(relu3, "pred", dim_out = 10)
loss = L.SoftmaxWithLoss(pred, "loss", label_data = data)

net = Net()
net.setLoss(loss)

net.lr = 0.2
```

The training and predicting codes are in `test/` folders, namely `digit_recognizerKaggle2.py` and `digit_recognizerKaggle2pred`.

Enjoy it! :-)
