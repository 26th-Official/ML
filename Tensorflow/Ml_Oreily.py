import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets\housing.csv")
print(data.head())
print(data.info())


data["income_category"] = pd.cut(data["median_income"],
                                 labels=[1,2,3,4,5],
                                 bins=[0.0,1.0,3.0,5.0,6.0,np.inf])

x_train,x_test = train_test_split(data,test_size=0.2,random_state=42,stratify=data["income_category"])
# print(x_train)
# print(x_test)
for i in(x_train,x_test):
    i.drop("income_category",axis=1,inplace=True)
# print(len(x_train))
# print(len(x_test))



# data.plot(kind="scatter", x="latitude",y = "longitude",grid=True,alpha=0.2)

# data.plot(kind="scatter",x="median_house_value",y="median_income",alpha=0.2)

t_data = x_train.copy()


t_data["room_per_house"] = t_data["total_rooms"]/t_data["households"]
t_data["bedrooms_ratio"] = t_data["total_bedrooms"]/t_data["total_rooms"]
t_data["people_per_house"] = t_data["population"]/t_data["households"]
print(t_data.corr()["median_house_value"].sort_values(ascending=False))


t_data = x_train.drop("median_house_value",axis=1)
t_data_label = x_train["median_house_value"].copy()
# print(t_data.head())
# print(t_data_label)

t_data_num = t_data.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy="median")
temp = imputer.fit_transform(t_data_num)


t_data_num = pd.DataFrame(temp,columns=t_data_num.columns,index=t_data_num.index)
print(t_data_num.info())

t_data_str = t_data[["ocean_proximity"]]
print(t_data_str)
# print(t_data["ocean_proximity"].unique())

ord_encoder = OrdinalEncoder()
temp = ord_encoder.fit_transform(t_data_str)
# print(temp)

t_data_str = pd.DataFrame(temp,columns=t_data_str.columns,index=t_data_str.index)
print(t_data_str)























plt.show()





