# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:16:12 2019

@author: Raval
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000 # no of data
d = 10 # no of ads or no of columns or no of arms

number_of_selection = [0] * d # vectoer of size d containing only zeroes.
# we are doing this because at the first round each version of the ad hasn't selecting yet,so no of time each ad is selected is 0
sum_of_rewards = [0] * d # sum of the reward of each version of the ad is 0.
ads_selected = [] # ad selected after computing UCB in each data
total_reward = 0

for n in range(0, N): # computing all the data.
    ad = 0
    max_upper_bound = 0
    for i in range(0, d): # computing all the ads in each data.
        if(number_of_selection[i]>0):
            avereage_reward = sum_of_rewards[i] / number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selection[i]) # we do (n + 1) because indexes in pytthon starts with 0. in the math formula log(n) but in here we use (n+1)
            # computing the UCB
            upper_bound = avereage_reward + delta_i
        else:
            upper_bound = 1e400 # 10 to the power of 400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i # we take the one or each data.
    ads_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad] + 1 
    # to take the reward from the dataset index is round ex. index 1 in 1 round index 2 is 2 round and other column is reward
    reward = dataset.values[n ,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
      
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
