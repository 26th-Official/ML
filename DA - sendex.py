import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("avocado.csv")
df = df[df["type"]=="organic"]
df["Date"] = pd.to_datetime(df["Date"])
# df.set_index("Date",inplace=True)
# df.sort_index(inplace=True)
print(df["region"].unique())

# graph_df = pd.DataFrame()

# for i in df["region"].unique()[:16]:
#     print(i)
#     region = df.copy()[df["region"] == i]
#     region.sort_index(inplace=True)
#     region[f"{i}_Avg-price"] = region["AveragePrice"].rolling(25).mean()

#     if graph_df.empty:
#         graph_df = region[[f"{i}_Avg-price"]]
#     else:
#         graph_df = graph_df.join(region[f"{i}_Avg-price"])

new = pd.DataFrame()

for name,group in df.groupby("region"):
    if new.empty:
        new = group.set_index("Date")[["AveragePrice"]].rename(columns={"AveragePrice":name})
    else:
        new = new.join(group.set_index("Date")[["AveragePrice"]].rename(columns={"AveragePrice":name}))

print(new)
new.to_csv("sam.csv")

# graph_df.dropna(inplace=True)   
# graph_df.to_csv("sam.csv")



# df.plot()
plt.show()
# print(df)







