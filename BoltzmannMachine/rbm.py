#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:21:18 2019

@author: shidiyang
??? Loss to high
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 

#Col0: users, Col1: Movie ID, Col2: ratings (1 to 5), col3: Timestamp
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Prepare the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', sep = '\t', header = None)
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', sep = '\t', header = None)
test_set = np.array(test_set, dtype = 'int')

#Getting the total number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#Converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_user] #select the data from col1 such that col0 == id_user
        id_ratings = data[:,2][data[:,0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings #rating starts from 0
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensors
## Tensors: multi-dim arrays that contain single type elements
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Converting the rating into binary rating 1 (like) and 0 (Unlike)
training_set[training_set == 0] = -1 #unrated
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Creating the architecture of the Neural Network
class RBM():
    """
    nv: the number of visible nodes
    nh: the number of hidden nodes
    """
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) #initial random weights 
        self.a = torch.randn(1, nh) #bias for hidden nodes
        self.b = torch.randn(1, nv) #bias for visible nodes
    
    """
    x: corresponding to the visible neurons v in the probabilities
    """
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) #product of to tensors
        activation = wx + self.a.expand_as(wx) #expand_as: the bias are applied to each line of the mini batch
        p_h_given_v = torch.sigmoid(activation) #probabilities that hidden node is activated given v
        
        #sample (yes 1/no 0) of the hidden neurons in this probablity
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #Contrastive Divergence
    """
    v0: input vector containin the ratings of all the movies by one user
    vk: visible nodes obtained after K samplings
    ph0: the vector of probabilities that at the first iteration the hidden nodes equal one given the values of v0
    phk: the vector of probabilities that at the k iteration the hidden nodes equal one given the values of vk
    """
    def train(self, v0, vk, ph0, phk):
        #update tensor of weights W
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        
        #update tensor of bias b
        self.b = torch.sum((v0 - vk), 0)
        
        #update tensor of bias a
        self.a = torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

#Training the RBN
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        
        #get 10th sample of hidden nodes and visible nodes
        for k in range(10):
            #gibb sampling
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here, vk: prediction
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users): #looping all the users in test set
    
    # training set is the input that will be used to activate the hidden neurons to get the output
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1] #target
    
    #1 step of blind walk
    if len(vt[vt>=0]) > 0:
        #gibb sampling
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here, vk: prediction
        s += 1. #normalize test loss
print('loss: '+str(test_loss/s))