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
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

# load data
batch_size = 256
iter_trainning, iter_testing = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# hyper parameter        
number_epochs = 5
d2l.train_ch3(net, iter_trainning, iter_testing, loss, number_epochs, batch_size, None, None, trainer)
