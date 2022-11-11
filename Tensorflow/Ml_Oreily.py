import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer,make_column_selector,make_column_transformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_array,check_is_fitted


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

class ClusterSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self,n_clusters = 10,gamma=1.0,random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self,x,y=None,sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters,random_state=self.random_state)
        self.kmeans_.fit(x,sample_weight=sample_weight)
        return self
    
    def transform(self,x):
        return rbf_kernel(x,self.kmeans_.cluster_centers_,gamma=self.gamma)
    
    def get_feature_names_out(self,names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
def column_ratio(x):
    return x[:,[0]]/x[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

cluster = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

ratio_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("column",FunctionTransformer(column_ratio,feature_names_out=ratio_name))
])

num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
])

log_pipeline = Pipeline([
    ("log",FunctionTransformer(np.log,feature_names_out="one-to-one"))
])

cat_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("ord_encoder", OrdinalEncoder())
])


preprocessing = ColumnTransformer([
    ("room_per_house",ratio_pipeline,["total_rooms","households"]),
    ("bedrooms",ratio_pipeline,["total_bedrooms","total_rooms"]),
    ("people_per_house",ratio_pipeline,["population","households"]),
    ("l_",log_pipeline,["total_rooms","total_bedrooms", "population", "households"
                       ,"median_income"]),
    ("k_",cluster,["longitude", "latitude"]),
    ("c_",cat_pipeline,make_column_selector(dtype_include=object))
])



print(t_data)
final_data = preprocessing.fit_transform(t_data)
print(preprocessing.get_feature_names_out())





























plt.show()





