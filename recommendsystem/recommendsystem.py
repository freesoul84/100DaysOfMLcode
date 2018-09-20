#recommend system
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# dataset importing
dataset = pd.read_csv('recommendsystem.csv', header = None)

dataset.head(10)
length=len(dataset)
transactions=[]
for row in range(0,length):
    for col in range(0,20):
        item=str(dataset.values[row,col])
        transactions.append(item)
        
print(transactions)

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

print(rules)
results = list(rules)
print(results)

lift = []
association = []
for i in range (0, len(results)):
    lift.append(results[:len(results)][i][2][0][3])
    association.append(list(results[:len(results)][i][0]))
    

#a =association ,l=lift
rank = pd.DataFrame([association, lift]).T
rank.columns = ['a', 'l']

# Show top 10 higher lift scores
rank.sort_values('l')
print("Association :",association)
print("rank :",rank)

