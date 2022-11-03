import pandas as pd
from sklearn import preprocessing,svm
import sklearn


df = pd.read_csv("Datasets/diamonds.csv",index_col=0)

cut_dict = {"Fair":1, "Good":2, "Very Good":3, "Premium":4, "Ideal":5}
color_dict = {"D":7,"E":6,"F":5,"G":4,"H":3,"I":2,"J":1}
clarity_dict = {"I1": 1,"SI2":2 ,"SI1": 3,"VS2":4 ,"VS1": 5,"VVS2": 6,"VVS1":7 ,"IF": 8}

df["cut"] = df["cut"].map(cut_dict)
df["color"] = df["color"].map(color_dict)
df["clarity"] = df["clarity"].map(clarity_dict)


df = sklearn.utils.shuffle(df)

x = df.drop("price",axis=1).values
y = df["price"].values

x = preprocessing.scale(x)

size = 200

x_train = x[:-200]
y_train = y[:-200]

x_test = x[-200:]
y_test = y[-200:]


classify = svm.SVR(kernel="linear")
classify.fit(x_train,y_train)

print(classify.score(x_test,y_test))

for i,j in zip(x_test,y_test):
    print(f'Test Data: {classify.predict([i])[0]}, Correct data: {j}')

