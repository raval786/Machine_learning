# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:55:04 2019

@author: Raval
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv' ,header=None)# we remove the header because the first row of every column is not header it shows the first item sell on there store

transaction = []
for i in range(0 , 7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])# dataset.values[i,j] i is the all the data set and j is for all the transcation
#for j in range(0,20) is for atking columns and we put [] bracket fot list from dataset to to end of the for loop to make a list 
# we put str str(dataset.values[i,j]) in list of dataset because apriory algorithm wants the string
    
from apyori import apriori
rules = apriori(transaction , min_support=0.003 , min_confidence=0.2 , min_lift=3 , min_length=2)# min_length= 2 we select 2 because it is minimum no of product
#product is buy 3 times a day. 7 for week and 7500 is no.of transcation
#  3*7/7500=0.0028  so 0.0028 is the minimum support is 0.003

# min_confidence we can't set it to high because if we put ex. 0.8 min_confidence we get obvious result so they can't associate well so we put 0.2 20% 
#min lift is equal or grater then 3 so we can get some great rule0

result = list(rules)
print(result)
ll =pd.DataFrame(result)
