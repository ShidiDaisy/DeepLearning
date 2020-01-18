#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:02:44 2020

@author: shidiyang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

data = pd.read_csv('sales_data_for_test.csv', index_col=0)
data = data.sort_values(by='INVDT', ascending=True)
data = data.set_index('INVDT')
filtered_data = data[(data.MRP_VALUE >= 0) & (data.NETSALE_VALUE >= 0) & (data.TAX_VALUE >= 0) & (data.SALE_QTY < 500)]

training_set = filtered_data[filtered_data.index < '2019-09-01']
test_set = filtered_data[filtered_data.index >= '2019-09-01']

#select the features
training_set_selected = training_set.iloc[:,2:10]
test_set_selected = test_set.iloc[:,2:10]

# normalize features
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set_selected.values)
test_set_scaled = sc.transform(test_set_selected.values)

# select features
training_set_scaled_X = training_set_scaled[:,[0,1,2,3,4,5,6,7]]
training_set_scaled_y = training_set_scaled[:,[2]]
test_set_scaled_X = test_set_scaled[:,[0,1,2,3,4,5,6,7]]
test_set_scaled_y = test_set_scaled[:,[2]]

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(7, 683286):
    X_train.append(np.concatenate((np.array([training_set_scaled_X[i],]*7), training_set_scaled_y[i-7:i]),axis = 1)) 
    y_train.append(training_set_scaled_y[i])
    
for i in range(7, 56130):
    X_test.append(np.concatenate((np.array([test_set_scaled_X[i],]*7), test_set_scaled_y[i-7:i]),axis = 1))
    y_test.append(test_set_scaled_y[i])
    
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 9)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1)) #output layer

#compile
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the rnn with training set
regressor.fit(X_train, y_train, epochs = 5)

#Save model
# regressor.save('classifier_model.h5', overwrite=True)


#predict
predicted_sales_qty= regressor.predict(X_test)

# Scaling Back
pred_arr = test_set_scaled[7:]
predicted_sales_qty = predicted_sales_qty.reshape(56123)
pred_arr[:,2] = predicted_sales_qty
predicted_df = sc.inverse_transform(pred_arr)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, predicted_sales_qty))
rmse_actual = sqrt(mean_squared_error(test_set.iloc[7:,2].values, predicted_df[:,2]))

predicted_df_2 = pd.DataFrame(predicted_df)
predicted_df_2.columns = ['MCAT', 'SUBCAT', 'SALE_QTY', 'MRP_VALUE', 'NETSALE_VALUE', 'TAX_VALUE',	'PRODUCT', 'SHOP']

product_df = predicted_df_2.groupby('PRODUCT').sum()
product_df_actual = test_set.iloc[7:,].groupby('PRODUCT').sum()
product_rmse = sqrt(mean_squared_error(product_df_actual['SALE_QTY'].values, product_df['SALE_QTY'].values))

subcat_df = predicted_df_2.groupby('SUBCAT').sum()
subcat_df_actual = test_set.iloc[7:,].groupby('SUBCAT').sum()
subcat_rmse = sqrt(mean_squared_error(subcat_df_actual['SALE_QTY'].values, subcat_df['SALE_QTY'].values))

mcat_df = predicted_df_2.groupby('MCAT').sum()
mcat_df_actual = test_set.iloc[7:,].groupby('MCAT').sum()
mcat_rmse = sqrt(mean_squared_error(mcat_df_actual['SALE_QTY'].values, mcat_df['SALE_QTY'].values))



