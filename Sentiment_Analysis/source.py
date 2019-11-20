'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


names = ['content', 'Class']

dataset = pd.read_csv('dataset.csv',names=names)

print(dataset.head(5))

X = dataset.iloc[0].values
y = dataset.iloc[1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scalar = StandardScaler()
scalar.fit(X_train)


X_train = scalar.tranform(X_train)
X_test = scalar.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

'''
import csv
import re
import xlrd
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
        #print(c,msg)
        #f=0
        #nf=0
        #if type(msg) is str:
            #if(c == 1):
                #classification = "fraud"
            #f += 1
            #else:
                #classification = "notfraud"
            #nf += 1
            #classificaton = c
        #example_words = word_tokenize(str(msg))
        #msg = str(msg)
        #clean = [word for word in msg.split() if word not in stop_words]
        #msg = " ".join(clean)
        #clean = [word for word in msg.split() if len(word) >= 3]
        #msg = " ".join(clean)
        
        #msg = ''.join(ch for ch in msg if ch not in exclude)
        #msg = " ".join(lemma.lemmatize(word) for word in msg.split())
        #msg = re.sub(r"\d+","",msg)
        #msg = msg.split()
        if (len(msg) > 0):
            #msg = word_tokenize(str(msg))
            #msg = str(msg)
            msg = " ".join(filter(lambda x:x[0]!='@', msg.split()))
            clean = [word for word in msg.split() if word not in stop_words]
            msg = " ".join(clean)
            clean = [word for word in msg.split() if len(word) >= 3]
            msg = " ".join(clean)
        
            msg = ''.join(ch for ch in msg if ch not in exclude)
            msg = " ".join(lemma.lemmatize(word) for word in msg.split())
            msg = re.sub(r"\d+","",msg)
            msg = msg.split()
            rows.append({'message':str(msg),'class':str(c)})#lassification})
            index.append('y')
            j += 1
        #print("output", f," ", nf)
    print("j ",j)
    return DataFrame(rows,index = index)
	
data = DataFrame({'message':{}, 'class':{}})
data = data.append(dataFrameFromDirectory())
print(data.head())
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
#classifier = MultinomialNB()
classifier = KNeighborsClassifier(n_neighbors = 13 ) #MultinomialNB()
print('13')
targets = data['class'].values
classifier.fit(counts, targets)

#examples = ['i am happy to see you here','i am very angry man','thank you very much','i am so scared','please forgive me','i love you','i am surprised','this is so disappointing','i am very sad','it was fun to have you here']
examples = ['i am happy to see you here','I am way to sleepy.. Ill watch my shows lata..Good nite twit-fam!.. God bless!..XoXo','thank you very much','i am so scared','please forgive me','i love you','i am surprised','i hate you','i am very sad','it was fun to have you here']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)

