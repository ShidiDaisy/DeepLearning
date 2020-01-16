#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:02:44 2020

@author: shidiyang
"""
import pandas as pd
data = pd.read_csv('sales_data_for_test.csv')
data = data.sort_values(by='INVDT', ascending=True)
data.plot(kind='bar', x='WEEK', y='SALE_QTY')
plt.show()