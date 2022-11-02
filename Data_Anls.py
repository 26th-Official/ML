from functools import cache
import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/Min_wage.csv")


exp = pd.DataFrame()
for name,group in df.groupby("State"):
    if exp.empty:
        exp = group.set_index("Year")[["Low.Value"]].rename(columns={'Low.Value':name})
    else:
        exp = exp.join(group.set_index("Year")[["Low.Value"]].rename(columns={"Low.Value":name}))

exp = exp.replace(0,np.NaN)
exp.dropna(axis=1,inplace=True)
# print(exp.head())

df1 = pd.read_csv("Datasets/Unemployment rate.csv")
# print(df1)


@cache

def compare(year,state):
    try:
        return exp.loc[year][state]
    except:
        return np.NaN

df1["min_wage"] = list(map(compare,df1["Year"],df1["State"]))
df1.dropna(inplace=True)
print(df1)

print(df1[["Rate","min_wage"]].corr())
print(df1[["Rate","min_wage"]].cov())