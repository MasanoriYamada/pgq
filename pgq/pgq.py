#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()


import chainer
import chainer.links as L
import chainer.functions as F
from chainerrl.action_value import DiscreteActionValue

class PgqDQN(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        self.obs_size = obs_size
        self.n_actions = n_actions
        super(PgqDQN, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(obs_size, 100)
            self.fc2 = L.Linear(100, n_actions)
            self.fc3 = L.Linear(100, 1)

    def __call__(self, x):
        h1 = F.relu(self.fc1(x))
        V = self.fc3(h1) # (batch, 1)

        A = self.fc2(h1)

        # PGQ
        pi = F.softmax(A, axis=-1)
        pi.unchain_backward()
        base = F.sum(A * pi, axis=1, keepdims=True) # (batch, 1)
        A = A -  F.tile(base, (1, self.n_actions)) # (batch, action)

        Q = A + F.tile(V, (1, self.n_actions)) # (batch, action)
        return DiscreteActionValue(Q)

