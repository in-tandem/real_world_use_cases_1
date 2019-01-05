import numpy as np 
import pandas as panda 

merchant_cat_id_added = panda.read_csv('all/merchant_cat_id_added.csv')

train_data = panda.read_csv('all/my_training_data_can_be_used_as_is.csv')
test_data = panda.read_csv('all/my_test_data_categorized_can_be_used_as_is.csv')

print(len(merchant_cat_id_added), len(merchant_cat_id_added.columns.tolist()))


