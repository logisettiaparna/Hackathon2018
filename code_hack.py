#importing required Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import nltk 
from collections import Counter
import matplotlib
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import re

#importing data

from pyhive import hive            
conn = hive.connect(host='dbslp0500', port=10002, username="Username" , password="Password",auth="LDAP")
cur = conn.cursor()
cur.execute("set mapred.job.queue.name=araadh_q1.aratrans_sq1")
df= pd.read_sql("select * from clinical_temp.data_final_hack1 where rownum <= 4000",conn)

#df_1 = df.loc[df['data_refine5to8.supervisor'].isin(['Allegra Warren                                                                                                                                                                                                                                                 '])]
df_1 = df[['data_final_hack1.comment','data_final_hack1.surveyid']]
df_1.columns = ['comment','surveyid']

text1

#Data Cleaning
text1=df_1["comment"].values.tolist()
tweet1 = ' '.join(text1)
tweet1=tweet1.lower()
#text2x = df_1.head(4000)
#text2=text2x["comment"].values.tolist()
#tweet1 = ' '.join(text2)
#tweet1=tweet1.lower()
nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')


#Data Cleaning
def data_cleaning(tweet,custom_list):
    tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
    tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
    #stop = set(stopwords.words('english'))
    tweet = re.sub(r'[^a-zA-Z0-9]'," ",tweet)
    stop_words=set(['a',    'about', 'above', 'after',   'again',  'against',              'ain',      'all',        'am',               'an',       'and',     'any',     'are',      'as',        'at',        'be',       'because',            'been',   'before',               'being',  'below', 'between',           'both',   'but',      'by',        'can',     'couldn',               'd',               'did',      'didn',    'do',       'does',   'doesn', 'doing',  'don',     'down',  'during',               'each',               'few',     'for',      'from',   'further',              'had',     'hadn',   'has',      'hasn',   'have',   'haven',               'having',               'he',       'her',      'here',    'hers',    'herself',              'him',     'himself',               'his',       'how',    'i',           'if',         'in',         'into',     'is',         'isn',       'it',         'its',        'itself',               'just',     'll',          'm',        'ma',      'me',      'mightn',              'more',  'most',   'mustn', 'my',               'myself',               'needn', 'now',    'o',         'of',        'off',      'on',       'once',   'only',    'or',               'other',  'our',      'ours',    'ourselves',          'out',      'over',    'own',    're',        's',          'same',               'shan',   'she',      'should',               'so',        'some',  'such',    't',          'than',    'that',    'the',               'their',   'theirs',  'them',  'themselves',      'then',    'there',  'these',  'they',    'this',     'those',               'through',            'to',        'too',     'under', 'until',    'up',       've',        'very',    'was',     'we',               'were',   'weren',               'what',   'when',  'where',               'which', 'while',  'who',    'whom',               'why',    'will',      'with',    'won',    'y',          'you',     'your',    'yours',  'yourself',               'yourselves'])
    exclude = set(string.punctuation)
    exclude1= set(custom_list)
    stop_words.update(exclude1)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in tweet.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

exclude_list=["twitter","uhc","pic","rt"]

tweet = data_cleaning(tweet1,exclude_list)


#Sentiment Analysis
def sentiment_function(text_column):
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank
    final_df=[]
    i=0
    for sentence in text_column:
#         print i
#         i=i+1
        tweet = re.sub(r'[^a-zA-Z0-9]'," ",sentence)
              
        tokenizer = treebank.TreebankWordTokenizer()
        pos_words = 0
        neg_words = 0
        tokenized_sent = [word.lower() for word in tokenizer.tokenize(tweet)]

        x = list(range(len(tokenized_sent))) # x axis for the plot
        y = []

        for word in tokenized_sent:
            if word in opinion_lexicon.positive():
                pos_words += 1
                y.append(1) # positive
            elif word in opinion_lexicon.negative():
                neg_words += 1
                y.append(-1) # negative
            else:
                y.append(0) # neutral

        if pos_words > neg_words:
            a= 'Positive'
        elif pos_words < neg_words:
            a= 'Negative'
        elif pos_words == neg_words:
            a= 'Neutral'
        from nltk.sentiment import SentimentIntensityAnalyzer
        vader_analyzer = SentimentIntensityAnalyzer()
        b = str(vader_analyzer.polarity_scores(sentence))
        final_df.append([sentence,a,b])
    return final_df

e = sentiment_function(text1) 

senti_df = pd.DataFrame(e,columns=["Comment","Sentiment","Polarity"])
senti_df.to_csv("file.csv")

# TOpic Clustering 
import numpy as np
import pandas as pd
import seaborn as sns
import nltk 
from collections import Counter
import matplotlib
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import re
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank

df = pd.read_csv("/mapr/users/neeti101/hack_data_2.csv")
df1 = df['comment']

## loop for tagging the parts of speech
for ii in range(0,len(df['comment'])):
    iz = df.loc[ii, 'comment']
    text = word_tokenize(iz)
    text = set(text)
    pos = nltk.pos_tag(text)
    c = []
    for i in pos:       
        if i[1] == 'NN':
            x=i[0]
            c.append(x)
            df.at[ii, "result"] = c

