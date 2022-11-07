import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Datasets/housing.csv")
print(data.head())
print(data.info())

x_train,x_test = train_test_split(data,test_size=0.2,random_state=42)
print(x_train)
print(x_test)