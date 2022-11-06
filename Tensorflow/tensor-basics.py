import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


train = pd.read_csv("Datasets/Titanic/train.csv")
test = pd.read_csv("Datasets/Titanic/eval.csv")

# print(train.head())
# print(test.head())

y_train = train.pop("survived")
y_test = test.pop("survived")

# print(y_train)
# print(y_test)

# print(train.head())
# print(test.head())

categorical = ["sex","class","deck","embark_town","alone"]
numerical =["age","n_siblings_spouses","parch","fare"]

features = []
for i in categorical:
    vocabulary = train[i].unique()
    features.append(tf.feature_column.categorical_column_with_vocabulary_list(i,vocabulary))

for i in numerical:
    features.append(tf.feature_column.numeric_column(i))


def input_func(data,label,shuffle=True,batch=32,epoch=10):
    def inp():
        data_in = tf.data.Dataset.from_tensor_slices((dict(data),label))
        if shuffle:
            data_in.shuffle(1000)
        data_in = data_in.batch(batch).repeat(epoch)
        return data_in
    return inp

train_inp = input_func(train,y_train)
test_inp = input_func(test,y_test,shuffle=False,epoch=1)

linear_est = tf.estimator.DNNClassifier(feature_columns=features,hidden_units=[30,10])
linear_est.train(train_inp,)
result = linear_est.evaluate(test_inp)
print(result["accuracy"])

final = list(linear_est.predict(test_inp))
print(final[0]["probabilities"])

# print(list(result)[0]["probablities"][1])
# plt.hist(train["age"],bins=20)
# plt.show()

