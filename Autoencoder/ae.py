#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:19:18 2019

@author: shidi
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

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        # (parentness), self: the object of SAE class
        super(SAE, self).__init__()
        
        #nb_movies: input parameters
        #20: the number of neurons
        self.fc1 = nn.Linear(nb_movies, 20)#first full connection