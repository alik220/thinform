import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import csv

#в file_train - то, на основе чего тренируем
file_train_open = open('news_train.txt', 'rt', encoding="utf8")
file = csv.reader(file_train_open, delimiter='\t')
file_train = list(file)

text_in_file = [i[2] for i in file_train]
key_word = [i[0] for i in file_train]

#tfidf - преобразование слов в частотах
baseon = Pipeline([('matr', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(alpha=1e-8, random_state=200))])

#строим модель
baseon.fit(text_in_file, key_word)

#в file_text - то, что классифицируем
file_text_open = open('news_test.txt', 'rt', encoding="utf8")
file2 = csv.reader(file_text_open, delimiter='\t')
file_text = list(file2)

text = [i[1] for i in file_text]

data_predict = baseon.predict(text)


out = open('file_final.txt', 'w')
for item in data_predict:
    out.write("%s\n" % item)