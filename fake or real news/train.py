import pandas as pd
from gensim.models.word2vec import Word2Vec
import string
import sys

input_file = sys.argv[1]
output_model = sys.argv[2]

train = pd.read_csv(input_file)
train.head()

fake_train = train[train["label"] == "FAKE"]
real_train = train[train["label"] == "REAL"]

fake_news = pd.DataFrame()
real_news = pd.DataFrame()

#fake_news["words"] = fake_train["text"]
fake_news["words"] = fake_train["title"] + fake_train["text"]
#real_news["words"] = real_train["text"]
real_news["words"] = real_train["title"] + real_train["text"]

fake_news["words"] = fake_news["words"].str.translate(str.maketrans("","", '!“"”#$%&\'’()*+,-./:;<=>?@[\\]^_`{|}~—'))
real_news["words"] = real_news["words"].str.translate(str.maketrans("","", '!“"”#$%&\'’()*+,-./:;<=>?@[\\]^_`{|}~—'))

fake_news["words"] = fake_news["words"].str.split(" |\\n")
real_news["words"] = real_news["words"].str.split(" |\\n")

fake_model = Word2Vec(fake_news["words"], min_count=7, size=300, iter=10, window=5)
real_model = Word2Vec(real_news["words"], min_count=7, size=300, iter=10, window=5)

fake_model.save(output_model+'/fake_model')
real_model.save(output_model+'/real_model')
