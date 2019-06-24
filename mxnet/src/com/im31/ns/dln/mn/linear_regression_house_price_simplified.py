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
 
from mxnet import autograd, nd
from mxnet.gluon import data as gluon_data
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gluon_loss
from mxnet import gluon

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

prices_example = weights_true[0] * features[:, 0] + weights_true[1] * features[:, 1] + bias_true
noise = nd.random.normal(loc=0, scale=0.01, shape=prices_example.shape)
prices_example += noise
      
# hyper parameter
batch_size = 5

dataset = gluon_data.ArrayDataset(features, prices_example)

data_iter = gluon_data.DataLoader(dataset, batch_size, shuffle=True)

# model
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))

# loss function
loss = gluon_loss.L2Loss()

# hyper parameter        
learning_rate = 0.0003

# optimization algorithm
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

# hyper parameter
number_of_epochs = 3000

for epoch in range(1, number_of_epochs + 1):
    for features_batch, prices_batch in data_iter:
        with autograd.record():
            loss_batch = loss(net(features_batch), prices_batch)
        loss_batch.backward()
        trainer.step(batch_size)
    loss_trainning = loss(net(features), prices_example)
    print('epoch %d, loss %f' % (epoch, loss_trainning.mean().asnumpy()))

dense = net[0]

weights_trainning = dense.weight.data()
bias_trainning = dense.bias.data()

print('weights_true ', weights_true)
print('weights_trainning ', weights_trainning)  

print('bias_true %f' % (bias_true))
print('bias_trainning ', bias_trainning) 
