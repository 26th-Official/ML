import pandas as pd
import math
import pickle
from sklearn import preprocessing,svm,neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

df = pd.read_csv("Datasets/Stocks.csv")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close',"Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"] * 100
df["PCT_CHG"] = (df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"] * 100


df = df[["Adj. Close","HL_PCT","PCT_CHG","Adj. Volume"]]
df.fillna(-99999,inplace=True)

# df.to_csv("mod1.csv")

forecast = "Adj. Close"
forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out)

df["Label"] = df[forecast].shift(-forecast_out)
df.dropna(inplace=True)
# df.to_csv("mod2.csv")

X = np.array(df.drop(["Label"],axis=1))
y = np.array(df["Label"])
X = preprocessing.scale(X)
# print(len(X))
# print(len(y))

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

classify = LinearRegression()
# classify = neighbors.KNeighborsClassifier()

classify.fit(X,y)
# with open("Linear_reg.pickle","wb") as f:
#     pickle.dump(classify,f)

# pickle_file = open("Linear_reg.pickle","rb")
# classify = pickle.load(pickle_file)

print(classify.score(x_test,y_test))

# for i,j in zip(x_test,y_test):
#     print(f"Test Data: {classify.predict([i])[0]} Original: {j}")

# df["Adj. Close"].plot()
# plt.show()


# print(df.head())


