import pandas as pd
import quandl
import math
from sklearn import preprocessing,svm,model_selection
import numpy as np

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close',"Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"] * 100
df["PCT_CHG"] = (df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"] * 100

# print(df["HL_PCT"],df["PCT_CHG"])

df = df[["Adj. Close","HL_PCT","PCT_CHG","Adj. Volume"]]

df.fillna(-99999,inplace=True)

forecast = "Adj. Close"

forecast_out = int(math.ceil(0.01*len(df)))
# print(len(df),forecast_out)

df["label"] = df[forecast].shift(-forecast_out)
df.dropna(inplace=True)

print(df.head())
