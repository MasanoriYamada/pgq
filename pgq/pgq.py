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
from chainerrl import distribution
from chainerrl.agents import a3c

class PgqDQN(chainer.Chain, a3c.A3CModel):
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
        # pi.unchain_backward()
        base = F.sum(A * pi, axis=1, keepdims=True) # (batch, 1)
        A = A -  F.tile(base, (1, self.n_actions)) # (batch, action)

        Q = A + F.tile(V, (1, self.n_actions)) # (batch, action)
        return DiscreteActionValue(Q)

    def pi_and_v(self, x):
        def V(x):
            h1 = F.relu(self.fc1(x))
            vout = self.fc3(h1) # (batch, 1)
            return vout
        def pi(x):
            h1 = F.relu(self.fc1(x))
            A = self.fc2(h1)
            piout = F.softmax(A, axis=-1)

            return distribution.SoftmaxDistribution(
            piout, beta=1.0, min_prob=0.0)
        return pi(x), V(x)


