import pandas as panda 
import numpy as np 

# this has 337 words..nltk has 179 stop words
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess # converts into lowercase tokens
from gensim import corpora

from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(2018)
# nltk.download('wordnet')
# read the complaint reason types
# lower case tokens
# stop word removal
# lemmetization - convert 3rd person to first person. and all tense to present tense
# stemming - get to root words


# create a bag of words
# input bag of words to LDA - get say ~75topics
# predict topic for each column and check accuracy


# create bag of words
# create tf idf 
# input tfidf to LDA - get say ~75 topics
# predict word for each column and check accuracy

### at the end we will have 75 topics for complaint reason in test as well as train
### simply replace the topic probability in the complaint reason column

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    lemetized =  [stemmer.stem(WordNetLemmatizer().lemmatize(i, pos='v')) for i in text]
    if len(lemetized) == 0:
        print('this gave me 0 text..it removed everything ', text)
    
    return lemetized

def stop_words_removed(text):
    removed =  [word for word in text if word not in STOPWORDS]
    if len(removed) == 0: # for a complaint reason like Others, stopword was removing this. we need others
        print('This removed all my words: ', text)
        removed = text
    
    return removed

train_data = panda.read_csv('dataset/train.csv')
train_data_complaint_reason_types = set(train_data['Complaint-reason'].values.tolist())

test_data = panda.read_csv('dataset/test.csv')
test_data_complaint_reason_types = set(test_data['Complaint-reason'].values.tolist())

print(train_data_complaint_reason_types)

# simple_preprocess by default removes workds lower than 2 letters and converts into lower case
tokenized_complaint_reason_types = [simple_preprocess(word) for word in train_data_complaint_reason_types]

print(tokenized_complaint_reason_types)

cleaned_complaint_reason_types = [stop_words_removed(word) for word in tokenized_complaint_reason_types ]

lemmetized_complaint_reason_types = [lemmatize_stemming(word) for word in cleaned_complaint_reason_types]


dictionary = corpora.Dictionary(lemmetized_complaint_reason_types)
corpus = [dictionary.doc2bow(word) for word in lemmetized_complaint_reason_types]

topic_count = 25

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = topic_count, id2word=dictionary, passes=15)

topics = ldamodel.print_topics()

for topic in topics:
    print('found topic: ',topic)

# print(lemmetized_complaint_reason_types, len(lemmetized_complaint_reason_types))

# combined = [' '.join(words) for words in lemmetized_complaint_reason_types]

# print(set(combined), len(set(combined)))


# for topic in train_data_complaint_reason_types:
# phrase = 'Problem with a purchase or transfer'
# bow = lemmatize_stemming(stop_words_removed(simple_preprocess(phrase))) 
# print(bow)
# a = ldamodel.get_document_topics(dictionary.doc2bow(bow))
# print(a)

count = 0
topics = []
for item in test_data_complaint_reason_types:
    bow = lemmatize_stemming(stop_words_removed(simple_preprocess(item))) 
    # print(bow)
    a = ldamodel.get_document_topics(dictionary.doc2bow(bow))
    arr = np.array(a)
    t1,t2 = arr.max(axis=0)
    print('prediction: ', bow, a, type(a), int(t1))
    count+=1
    topics.append(int(t1))

print(count, len(test_data_complaint_reason_types),set(topics),len(set(topics)))





count = 0
topics = []
for item in train_data_complaint_reason_types:
    bow = lemmatize_stemming(stop_words_removed(simple_preprocess(item))) 
    # print(bow)
    a = ldamodel.get_document_topics(dictionary.doc2bow(bow))
    arr = np.array(a)
    t1,t2 = arr.max(axis=0)
    # print('prediction: ', bow, a, type(a), int(t1))
    count+=1
    topics.append(int(t1))

print(count, len(train_data_complaint_reason_types),set(topics),len(set(topics)))


train_complaint_reason = train_data['Complaint-reason'].values.tolist()
test_complaint_reason = test_data['Complaint-reason'].values.tolist()

tfidf = TfidfVectorizer()
# tfidf.fit(train_complaint_reason)
train_vector = tfidf.fit_transform(train_complaint_reason)

tfidf_1 = TfidfVectorizer(vocabulary = tfidf.vocabulary_)

test_vector = tfidf_1.fit_transform(test_complaint_reason)

print('tfidf',test_vector.toarray().shape, train_vector.shape, test_data.shape, train_data.shape)

