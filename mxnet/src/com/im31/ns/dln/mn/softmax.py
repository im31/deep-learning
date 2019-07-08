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
from mxnet import autograd, nd

r"""
Model: (input: 4 features, output: 3 kinds)

o1 = x1 * w11 + x2 * w21 + x3 * w31 + x4 * w41 + b1
o2 = x1 * w12 + x2 * w22 + x3 * w32 + x4 * w42 + b2
o3 = x1 * w13 + x2 * w23 + x3 * w33 + x4 * w43 + b3

O = [o1 o2 o3]

X = [x1 x2 x3 x4]

W = |w11 w12 w13 w14|
    |w21 w22 w23 w24|
    |w31 w32 w33 w34|
    |w41 w42 w43 w44|

b = [b1 b2 b3]

Y = softmax(O)
   
"""

# load data
batch_size = 256
iter_trainning, iter_testing = d2l.load_data_fashion_mnist(batch_size)

# initialize parameters
number_inputs = 784
number_outputs = 10
W = nd.random.normal(scale=0.01, shape=(number_inputs, number_outputs))
b = nd.zeros(number_outputs)

W.attach_grad()
b.attach_grad()

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(axis = 1, keepdims = True)
    return x_exp / partition

# model
def net(x):
    return softmax(nd.dot(x.reshape(-1, number_inputs), W) + b)

# loss (cross entropy)
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

def accuracy(y_hat, y):
    return (y_hat.argmax(axis = 1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis = 1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n

# hyper parameter        
number_epochs = 5
learning_rate = 0.1

def train(net, iter_trainning, iter_testing, loss, number_epochs, batch_size,
          parameters = None, leanrning_rate = None, trainer = None):
    for epoch in range(number_epochs):
        train_l_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        for X, y in iter_trainning:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(parameters, leanrning_rate, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += (y_hat.argmax(axis = 1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(iter_testing, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        
train(net, iter_trainning, iter_testing, cross_entropy, number_epochs, batch_size, [W, b], learning_rate)

for X, y in iter_testing:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis = 1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

stop = True

