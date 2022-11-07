import tensorflow as tf
import pandas as pd


data = pd.read_csv("Datasets/Iris.csv",index_col=False)
data.pop("Id")
print(data.head())

data["Species"].replace(data["Species"].unique(),[0,1,2],inplace=True)

train = data[:125]
test = data[125:]

y_train = train.pop("Species")
y_test = test.pop("Species")

print(y_train)

features = []
for i in data.keys():
    features.append(tf.feature_column.numeric_column(i,dtype=tf.float32))
print(features)

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


classifier = tf.estimator.DNNClassifier(
    feature_columns=features,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train, y_train, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test,y_test, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


