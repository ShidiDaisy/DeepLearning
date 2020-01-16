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
movies = pd.read_csv('../BoltzmannMachine/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('../BoltzmannMachine/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 

#Col0: users, Col1: Movie ID, Col2: ratings (1 to 5), col3: Timestamp
ratings = pd.read_csv('../BoltzmannMachine/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Prepare the training set and test set
training_set = pd.read_csv('../BoltzmannMachine/ml-100k/u1.base', sep = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('../BoltzmannMachine/ml-100k/u1.test', sep = '\t')
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
        self.fc2 = nn.Linear(20, 10) #10 neurons in second layer
        self.fc3 = nn.Linear(10, 20) #decode in third layer
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x)) #fc1 connected the vector at the left
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)#Try different optimizers (RMSprop, Adam)

# Train SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0. #count the users who have rated movie, later will be used in calculating RMSE
    for id_user in range(nb_users):
        #all the ratings for all the movies
        # training_set[id_user] is 1 dimension vector, but Pytorch doesn't accept single vector
        input = Variable(training_set[id_user]).unsqueeze(0) #0: create a batch of single input vector
        target = input.clone() 
        if torch.sum(target.data > 0) > 0: #user rated at least 1 movie
            output = sae(input) #forward() will be executed
            
            #Optimization
            target.require_grad = False # Don't compute gradient by target to save computation
            output[target == 0] = 0 # dont count
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            
            #backpropagation, only used in training
            loss.backward() #decide the direction to which the weight will be updated, decreased or increased
            
            #update train loss
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            
            #decides intensity of the updates, the amount of the weights will be updated, related to BP, 
            #only used in training
            optimizer.step() 
    print('epoch: '+ str(epoch) + ' loss: ' + str(train_loss/s))
    
# Testing the SAE
test_loss = 0
s = 0. #count the users who have rated movie, later will be used in calculating RMSE
for id_user in range(nb_users):
    #train the model based on the movies that rated by users, predict the movies that not rated
    input = Variable(training_set[id_user]).unsqueeze(0) #0: create a batch of single input vector
    
    #target is the real answer
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0: #user rated at least 1 movie
        output = sae(input) #forward() will be executed
        
        #Optimization
        target.require_grad = False # Don't compute gradient by target to save computation
        output[target == 0] = 0 # dont count
        loss = criterion(output,target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        
        #update train loss
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.

#the loss should be below 1 star -> the difference between prediction and the real is less than 1 star
print('loss: ' + str(test_loss/s)) 