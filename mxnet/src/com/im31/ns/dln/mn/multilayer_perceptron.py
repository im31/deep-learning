# Copyright (c) 2019, Xiao Wang.
# All rights reserved.
#
# This file is part of Deep Learning Note (DLN). see <https://github.com/im31/deep-learning-note>
#
# DLN is free software; you can redistribute it and/or modify it.
#
# DLN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# Contributors:
#     Xiao Wang - initial implementation

import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

# load data
batch_size = 256
iter_trainning, iter_testing = d2l.load_data_fashion_mnist(batch_size)

# initialize parameters
number_inputs = 784
number_outputs = 10
number_hiddens = 256

W1 = nd.random.normal(scale=0.01, shape=(number_inputs, number_hiddens))
b1 = nd.zeros(number_hiddens)

W2 = nd.random.normal(scale=0.01, shape=(number_hiddens, number_outputs))
b2 = nd.zeros(number_outputs)

parameters = [W1, b1, W2, b2]

for parameter in parameters:
    parameter.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, number_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

loss = gloss.SoftmaxCrossEntropyLoss()

# hyper parameter        
number_epochs = 5  
learning_rate = 0.5      
d2l.train_ch3(net, iter_trainning, iter_testing, loss, number_epochs, batch_size, parameters, learning_rate)
