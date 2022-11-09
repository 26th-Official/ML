import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import rbf_kernel


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

# print(t_data.corr()["median_house_value"].sort_values(ascending=False))


t_data = x_train.drop("median_house_value",axis=1)
t_data_label = x_train["median_house_value"].copy()

t_data["room_per_house"] = t_data["total_rooms"]/t_data["households"]
t_data["bedrooms_ratio"] = t_data["total_bedrooms"]/t_data["total_rooms"]
t_data["people_per_house"] = t_data["population"]/t_data["households"]

print(t_data.info())


# # print(t_data.head())
# print(t_data_label)

# t_data_num = t_data.select_dtypes(include=[np.number])

# t_data_str = t_data[["ocean_proximity"]]
# print(t_data_str)
# print(t_data["ocean_proximity"].unique())

# ord_encoder = OrdinalEncoder()
# temp = ord_encoder.fit_transform(t_data_str)
# print(temp)
# t_data_str = pd.DataFrame(temp,columns=t_data_str.columns,index=t_data_str.index)
# print(t_data_str)

# print(t_data[["housing_median_age"]].max())

# t_data.hist()


num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
])



cat_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("ord_encoder", OrdinalEncoder())
])

preprocessing = ColumnTransformer([
    ("int",num_pipeline,make_column_selector(dtype_include=np.number)),
    ("str",cat_pipeline,make_column_selector(dtype_include=object))
])


# t_data_num = pd.DataFrame(temp,columns=t_data_num.columns,index=t_data_num.index)
# print(t_data_num.info())

print(t_data)
# final_data = preprocessing.fit_transform(t_data)

# final_data = pd.DataFrame(final_data,columns=t_data.columns,index=t_data.index)
# print(final_data.info())
# print(final_data.head())

t_data[["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]].hist()


from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(t_data[["population"]])

























plt.show()





