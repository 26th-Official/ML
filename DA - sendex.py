import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("avocado.csv")
df["Date"] = pd.to_datetime(df["Date"])

region = df[df["region"]== "Albany"]

region.set_index("Date",inplace=True)
region.sort_index(inplace=True)
region["AveragePrice"].rolling(25).mean().plot()


plt.show()
