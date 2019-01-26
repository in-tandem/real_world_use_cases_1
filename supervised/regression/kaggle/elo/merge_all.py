import pandas as panda
import numpy as np 

train_data = panda.read_csv('all_new/train.csv')
new_data = panda.read_csv('all_new/treated_new_transaction_2.csv')

t = panda.concat([train_data,new_data],axis=1)
t.dropna(inplace= True)

