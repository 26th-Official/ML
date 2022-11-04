from nltk.corpus import stopwords,state_union,wordnet
from nltk.tokenize import word_tokenize,PunktSentenceTokenizer
from nltk.stem import PorterStemmer
import nltk

sys = wordnet.synsets("Hello")

print(sys[0].lemmas()[0].name())
print(sys[0].definition())

s = []
a = []

for i in sys:
    for j in i.lemmas():
        s.append(j.name())
        if j.antonyms():
            a.append(j.antonyms()[0].name())

print(a)
print(s)




# s = "Helloo, How are you doing? . I am doing fine."
# stop = set(stopwords.words("english"))

# word = word_tokenize(s)
# print(word)
# filtered = [i for i in word if i not in stop]
# print(filtered)

# ex = ["python","Pythoner","Pythone","pythonly"]
# ps = PorterStemmer()
# [print(ps.stem(i)) for i in ex]

# train = state_union.raw("2005-GWBush.txt")
# test = state_union.raw("2006-GWBush.txt")

# custom = PunktSentenceTokenizer(train)
# tokenized = custom.tokenize(test)
# # print(tokenized)

# for i in tokenized:
#     word = nltk.word_tokenize(i)
#     print(word)
#     tagged = nltk.pos_tag(word)
#     print(tagged)