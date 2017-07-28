#coding=utf-8
import numpy as np
from .LRUpdater import *

class Solver(object):
    def __init__(self, *args, **kwargs):
        self.iter_num = 0
        self.lr = 1.0
        self.lr_updater = LRUpdater(*args, **kwargs)
    def update(self, l):
        pass
    def init(self, l):
        # init each layer l
        pass
    @property
    def base_lr(self):
        return self.lr_updater.base_lr
    @base_lr.setter
    def base_lr(self, value):
        self.lr_updater.base_lr = value
    def update_lr(self, iter_num):
        self.lr = self.lr_updater.get_lr(iter_num)
    @property
    def lr_policy(self):
        return self.lr_updater.lr_policy
    @lr_policy.setter
    def lr_policy(self, value):
        self.lr_updater.set_policy(value)
