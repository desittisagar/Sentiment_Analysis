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
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
import itertools

contractions = {
"u": "you",
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

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
        msg = str(msg)
        if (len(msg) > 0):
            for word in msg.split():
                if word.lower() in contractions:
                    msg = msg.replace(word, contractions[word.lower()])
            #msg = " ".join(msg)
            msg = " ".join(re.findall('[A-Z][^A-Z]*', msg))
            #msg = slang_lookup(msg)
            msg = ''.join(''.join(s)[:2] for _, s in itertools.groupby(msg))     # standardising words i looovvvveeeee you to i love you
            
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
classifier = KNeighborsClassifier(n_neighbors = 13 ) #MultinomialNB()
print('13')
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['i am happy to see you here','Congratulations to phil packer on completing the \
            london marathon x a shining example to us all x',\
            'sorry, i am busy','thank you very much','i am so scared',\
            'please forgive me','i love you','i hate you','i am very sad',\
            'Hahaha @Jordan23Capp yes dey dooo, BOSTON Legal  tha fat old man is funny,\
            tha one that was naked in a pink gown Lol',\
            'HAPPY MOTHERS DAY TO ALL THE MOMMYS!!!!!!!!!!!!']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)