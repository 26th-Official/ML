import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets/housing.csv")
print(data.head())
# print(data.info())


data["income_category"] = pd.cut(data["median_income"],
                                 labels=[1,2,3,4,5],
                                 bins=[0.0,1.0,3.0,5.0,6.0,np.inf])

x_train,x_test = train_test_split(data,test_size=0.2,random_state=42,stratify=)
# print(x_train)
# print(x_test)

# print(len(x_train))
# print(len(x_test))





