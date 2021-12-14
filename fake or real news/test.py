import pandas as pd
from gensim.models.word2vec import Word2Vec
import string
import sys

input_file = sys.argv[1]
input_model = sys.argv[2]
query_file = sys.argv[3]

train = pd.read_csv(input_file)
train.head()

fake_train = train[train["label"] == "FAKE"]
real_train = train[train["label"] == "REAL"]

fake_news = pd.DataFrame()
real_news = pd.DataFrame()

fake_news["words"] = fake_train["text"]
#fake_news["words"] = fake_train["title"] + fake_train["text"]
real_news["words"] = real_train["text"]
#real_news["words"] = real_train["title"] + real_train["text"]

fake_news["words"] = fake_news["words"].str.translate(str.maketrans("","", '!“"”#$%&\'’()*+,-./:;<=>?@[\\]^_`{|}~—'))
real_news["words"] = real_news["words"].str.translate(str.maketrans("","", '!“"”#$%&\'’()*+,-./:;<=>?@[\\]^_`{|}~—'))

fake_news["words"] = fake_news["words"].str.split(" |\\n")
real_news["words"] = real_news["words"].str.split(" |\\n")

fake_model = Word2Vec.load(input_model+'/fake_model')
real_model = Word2Vec.load(input_model+'/real_model')

def top_5_similar(model, words, top=5):
    output = pd.DataFrame()
    for word in words:
        result = pd.DataFrame(model.wv.most_similar(word, topn=top), columns=[word, 'cosine similarity'])
        output = pd.concat([output, result], axis=1)
    return output

with open(query_file) as f:
    query_words = f.read().splitlines()

fake_result = top_5_similar(fake_model, query_words)
real_result = top_5_similar(real_model, query_words)

print("Result from FAKE news:\n", fake_result)
print("Result from REAL news:\n", real_result)