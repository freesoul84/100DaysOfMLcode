#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:13:30 2018

@author: alkesha
"""

import numpy as np
from apyori import apriori
import matplotlib.pyplot as mlt
import pandas as pd

data=pd.read_csv("market.csv",header=None)
print(data)
print(data.shape)
records=[]
for i in range(0,7501):
    records.append([str(data.values[i,j]) for j in range(0,20)])

print(records)

association_rules=apriori(records,min_support=0.0056,min_confidence=0.2,min_lift=3,min_length=3)
association_results=list(association_rules)
print(association_results)
print(len(association_results))

print(association_results[0])



for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
