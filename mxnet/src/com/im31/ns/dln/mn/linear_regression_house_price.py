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
 
import random

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd

r"""
Model:
label = weights_trainning * features + bias_trainning + noise
label = [w0, w1] * [f0, f1]T + bias_trainning = w0 * f0 + w1 * f1 + bias_trainning + noise
label = [f0, f1] * [w0, w1]T + bias_trainning = f0 * w0 + f1 * w1 + bias_trainning + noise
"""

number_of_features = 2
number_of_examples = 1000

r"""
house price: 2.5 * 10000 Yuan per square meter
depreciation according to the house age: 2 * 10000 Yuan per year
"""
weights_true = [2.5, -2.0]
bias_true = 2.2
features_area = nd.random.normal(loc=80, scale=10, shape=(number_of_examples, 1))
features_age = nd.random.normal(loc=6, scale=1, shape=(number_of_examples, 1))
features = nd.concat(features_area, features_age, dim=1)

r"""
Array features looks like:
NDArray: 
[[91.63078    5.84036  ]
 [84.83804    7.7974477]
 [82.995636   6.1959496]
 ...
 [79.13999    5.4559383]
 [81.68851    6.5341153]
 [96.51969    5.7332535]]
<NDArray 1000x2 @cpu(0)>
"""

prices_example = weights_true[0] * features[:, 0] + weights_true[1] * features[:, 1] + bias_true
noise = nd.random.normal(loc=0, scale=0.01, shape=prices_example.shape)
prices_example += noise

r"""
Array noise looks like:
NDArray: 
[ 1.46422136e+00 -1.30581355e+00  9.34440196e-01  5.38086295e-01
 -1.60380110e-01  8.41876030e-01 -1.00553632e+00  3.13221502e+00
 -5.28613091e-0...
 
 Array prices_example looks like:
 NDArray: 
[221.06046  197.3944   198.23163  198.03902  160.72455  229.70203
 174.89375  133.52571  205.08861  255.68204  224.8117   178.59067
...
"""

r"""
Plot a graph to show the relation between house price and house area. 
"""
display.set_matplotlib_formats('svg')
plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.scatter(features[:, 0].asnumpy(), prices_example.asnumpy(), 1)

def data_iter(batch_size, features, prices_example):
    number_of_examples = len(features)
    indices = list(range(number_of_examples))
    random.shuffle(indices)
    for i in range(0, number_of_examples, batch_size):
        index_segment = nd.array(indices[i: min(i + batch_size, number_of_examples)])
        yield features.take(index_segment), prices_example.take(index_segment)
        
# hyper parameter
batch_size = 5

weights_trainning = nd.random.normal(loc=0.0, scale=0.1, shape=(number_of_features, 1))
bias_trainning = nd.zeros(shape=(1,))

weights_trainning.attach_grad()
bias_trainning.attach_grad()

# model equation
def calculate_model_value(features, weights, bias):
    result = nd.dot(features, weights) + bias
    return result

# loss function
def squared_loss(values, labels):
    result = (values - labels.reshape(values.shape)) ** 2 / 2
    return result

# optimization algorithm
def stochastic_gradient_descent(parameters, learning_rate, batch_size):
    for parameter in parameters:
        parameter[:] = parameter - learning_rate * parameter.grad / batch_size

# hyper parameter        
learning_rate = 0.0003

# hyper parameter
number_of_epochs = 3000

for epoch in range(number_of_epochs):
    for features_batch, prices_batch in data_iter(batch_size, features, prices_example):
        with autograd.record():
            loss_batch = squared_loss(
                calculate_model_value(features_batch, weights_trainning, bias_trainning), prices_batch)
        loss_batch.backward()
        stochastic_gradient_descent([weights_trainning, bias_trainning], learning_rate, batch_size)
    best_prices = calculate_model_value(features, weights_trainning, bias_trainning)
    loss_trainning = squared_loss(
        calculate_model_value(features, weights_trainning, bias_trainning), prices_example)
    print('epoch %d, loss %f' % (epoch + 1, loss_trainning.mean().asnumpy()))

print('weights_true ', weights_true)
print('weights_trainning ', weights_trainning)  

print('bias_true %f' % (bias_true))
print('bias_trainning ', bias_trainning) 
