import csv
import re
import xlrd
import numpy
from bs4 import BeautifulSoup
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
from textblob import TextBlob
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

def cleanText(msg, lemmatize, stemmer):
    if isinstance(msg, float):
        msg = str(msg)
    if isinstance(msg, numpy.int64):
        msg = str(msg)
    try:
        msg = msg.decode()
    except AttributeError:
        pass

    soup = BeautifulSoup(msg, "lxml")
    msg = soup.get_text()                     #to get text from soup
    msg = re.sub(r"[^A-Za-z]", " ", msg)      #relace non alphabetical characters with spaces
    msg = msg.lower()                         #to lowercase
    msg = " ".join(filter(lambda x:x[0]!='@', msg.split()))
    clean = [word for word in msg.split() if word not in stop_words]
    msg = " ".join(clean)
    clean = [word for word in msg.split() if len(word) >= 3]
    msg = " ".join(clean)
    msg = ''.join(ch for ch in msg if ch not in exclude)
    msg = " ".join(lemma.lemmatize(word) for word in msg.split())
    msg = re.sub(r"\d+","",msg)
    text_result = []
    tokens = word_tokenize(msg)
    snowball_stemmer = SnowballStemmer('english')  #for lemmatization
    for t in tokens:
        text_result.append(snowball_stemmer.stem(t))
    return text_result

def dataFrameFromDirectory():
    rows = []
    index = []
    loc = "data1.xls"
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    j = 0
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    for i in range(n):
        c = sheet.cell_value(i,1)
        msg = sheet.cell_value(i,0)
        if (len(msg) > 0):
            msg = cleanText(str(msg), lemmatize = True, stemmer = False)
            rows.append({'message':str(msg),'class':str(c)})  #classification})
            index.append('y')
            j += 1
    print("j ",j)
    return DataFrame(rows,index = index)
	
data = DataFrame({'message':{}, 'class':{}})
data = data.append(dataFrameFromDirectory())
print(data.head())
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
#classifier = MultinomialNB()
classifier = KNeighborsClassifier(n_neighbors = 15 ) #MultinomialNB()
print('15')
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['i am happy to see you here','Congratulations to phil packer on completing the \
            london marathon x a shining example to us all x',\
            'sorry, i am busy','thank you very much','i am so scared',\
            'please forgive me','i love you','i hate you','i am very sad',\
            'Hahaha @Jordan23Capp yes dey dooo, BOSTON Legal  tha fat old man is funny,\
            tha one that was naked in a pink gown Lol',\
            'HAPPY MOTHERS DAY TO ALL THE MOMMYS!!!!!!!!!!!!']
for example in examples:
    example = cleanText(str(example), lemmatize = True, stemmer = False)

example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)