#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import svm

import pickle
#import os
#print(os.getcwd())



owner_model = pickle.load(open('./data/owner_model.pkl', "rb"))

print("* The USF LawnMower Owner Prediction Model *")
print("*****************************************************\n")
Income=float(input("Enter the income:"))
Lot_Size=float(input("Enter the lot size:"))
df = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})


result = owner_model.predict(df) 
probability = owner_model.predict_proba(df)
ownership = ('Owner', 'Non-Owner')
print(f"\nThe USF LawnMower Owner Prediction model indicates probability of ownership at {probability[0][1]:.4f}, therefore it implies that the person is {result[0]}.\n")


# In[ ]:




