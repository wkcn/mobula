#coding=utf-8
import copy
class Net:
    def __init__(self):
        self.loss = [] 
    def __str__(self):
        return "It is a network"
    def setLoss(self, l):
        if type(l) != list:
            l = [l]
        self.loss = copy.copy(l)
