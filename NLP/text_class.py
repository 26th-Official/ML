import json
from nltk.corpus import movie_reviews
import nltk
import random

d = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        d.append((list(movie_reviews.words(fileid)),category))

# for category in movie_reviews.categories():
#     print(category)
#     for fileid in movie_reviews.fileids(category):
#         temp = []
#         for k in movie_reviews.words(fileid):
#             if k.isalpha() == True and len(k) >1:
#                 temp.append(k)
#         d.append((temp,category))
#         temp = []

random.shuffle(d)
# print(d[10])


all_words = [i.lower() for i in movie_reviews.words() if i.isalpha() and len(i)>1]
# print(len(all_words))

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))


words_list = list(all_words.keys())[:3000]


def find_features(doc):
    words = set(doc)
    feature = {}
    for i in words_list:
        feature[i] = (i in words)
    return feature

# a = find_features(movie_reviews.words("neg/cv000_29416.txt"))
# print(a)

featureset = []
for i,j in d:
    featureset.append((find_features(i),j))
    print(featureset)
    print("=============================================================================")

# print(featureset)

train = featureset[1:5]
test = featureset[5:9]

with open("train.json","w") as f:
    json.dump(train,f)


with open("test.json","w") as v:
    json.dump(test,v)

# print(train[0][:1][0])
print("=============================================================================")
# print(test[0][5:10])

# classifier = nltk.NaiveBayesClassifier.train(train)
# print(f"Naive Bayes Accuracy - {(nltk.classify.accuracy(classifier,test))*100}")
# classifier.show_most_informative_features(15)



# true = 0
# false = 0
# for i in a.values():
#     if i == True:
#         true += 1
#     else:
#         false +=1


# print(f"True - {(true/(true+false))*100}%")
# print(f"False - {(false/(true+false))*100}%")





    