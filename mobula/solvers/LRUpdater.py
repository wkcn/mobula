#coding=utf-8
import numpy as np

# TODO: MULTISTEP
LR_POLICY_NUM = 7
class LR_POLICY:
    FIXED, STEP, EXP, INV, MULTISTEP, POLY, SIGMOID = range(LR_POLICY_NUM)

class LRUpdater:
    def __init__(self, *args, **kwargs):
        if "base_lr" in kwargs: 
            self.base_lr = kwargs["base_lr"] 
        else:
            self.base_lr = kwargs.get("lr", 1.0)
        self.gamma = kwargs.get("gamma", None)
        self.stepsize = kwargs.get("stepsize", None) 
        self.power = kwargs.get("power", None) 
        self.max_iter = kwargs.get("max_iter", None) 
        self.method = None
        self.lr_policy = kwargs.get("lr_policy", LR_POLICY.FIXED)
        self.set_policy(self.lr_policy)
    def set_policy(self, p):
        self.lr_policy = p
        self.method = LRUpdater.METHODS[p]
    def get_lr(self, iter_num):
        return self.method(self, iter_num)
    def fixed(self, iter_num):
        return self.base_lr
    def step(self, iter_num):
        # gamma, stepsize
        return self.base_lr * np.power(self.gamma, (iter_num // self.stepsize))
    def exp(self, iter_num):
        # gamma
        return self.base_lr * np.power(self.gamma, iter_num)
    def inv(self, iter_num):
        # gamma, power
        return self.base_lr * np.power(1.0 + self.gamma * iter_num, -self.power)
    def poly(self, iter_num):
        # power, max_iter
        return self.base_lr * np.power(1 - iter_num * 1.0 / self.max_iter, self.power)
    def sigmoid(self, iter_num):
        # gamma, stepsize
        return self.base_lr * (1.0 / (1.0 + np.exp(-self.gamma * (iter_num - self.stepsize))))

LRUpdater.METHODS = [None] * LR_POLICY_NUM
LRUpdater.METHODS[LR_POLICY.FIXED] = LRUpdater.fixed
LRUpdater.METHODS[LR_POLICY.STEP] = LRUpdater.step
LRUpdater.METHODS[LR_POLICY.EXP] = LRUpdater.exp
LRUpdater.METHODS[LR_POLICY.INV] = LRUpdater.inv
LRUpdater.METHODS[LR_POLICY.POLY] = LRUpdater.poly
LRUpdater.METHODS[LR_POLICY.SIGMOID] = LRUpdater.sigmoid
