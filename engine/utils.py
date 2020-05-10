import pandas as pd
import numpy as np
import collections, re
import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
from datetime import datetime

global nlp
nlp = spacy.load('es_core_news_sm')

def load_data(paths):
    """Load a list with dataframes containing the info of the csv files given as argument
    
    Key arguments:
    list paths -- list of paths to the data files
    
    Returns:
    list dataframes -- list of dataframes with the loaded data
    dataframe -- full combination of all dataframes
    """
    class_label = len(paths)-1
    index = 0
    dataframes = []
    tmp_column = []
    for path in paths:
        dataframes.append(pd.read_csv(path,
                                     delimiter = '|', 
                                     names = ['date', 'source', 'title', 'text']))
        x, y = dataframes[index].shape
        for i in range (0, x): tmp_column.append(class_label)

        dataframes[index]['class'] = tmp_column
        tmp_column = []
        index += 1
        class_label -= 1
        
    return dataframes, pd.concat(dataframes)

    
def preprocessing(dataset):
    """ Function that cleans the dataset given. Transforms the text to lowercase, removes punctuations,
    the 10 most frequent words and the 10 less frequent words.
    
    Arguments:
    DataFrame dataset -- DataFrame containing the instances and features.
    
    Returns:
    DataFrame dataset -- clean DataFrame containing the instances and features.
    """
    
    print (str(datetime.now())+" -- INFO: PREPROCESSING RUNNING\n")
    
    # Text to lowercase
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove punctuations
    dataset['text'] = dataset['text'].str.replace('[^\w\s]','')
    
    # Remove numerics
    dataset['text'] = dataset['text'].str.replace('[0-9]*','')
    
    # Remove stop-words
    stop_words = stopwords.words('spanish')
    newStopWords = ['si', 'sin', 'han', 'qué', 'que', 'cómo', 'como', 'cuando', 'cuándo', 'porque', 'no', 'loading', 
                    'día', 'dia', 'año']
    stop_words.extend(newStopWords)
    stop_words = set(stop_words)
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    
    # 10 most frequent words
    freq = pd.Series(' '.join(dataset['text']).split()).value_counts()[:10]
    freq = list(freq.index)
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    
    # 10 less frequent words
    freq = pd.Series(' '.join(dataset['text']).split()).value_counts()[-10:]
    freq = list(freq.index)
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    
    return dataset

def tokenize_2(text):
    tokens = word_tokenize(text)
    return tokens
    
def tokenize(text):
    lemmas = []
    for token in nlp(text):
        # Discard verbs and auxiliary verbs
        if (token.pos_ != 'VERB' and token.pos_ != 'AUX'):
            lemmas.append(token.lemma_)
    return lemmas

def calculateTfidf(instances):
    #removes features with frequency less than 5
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.5)
    response = vectorizer.fit_transform(instances)
    
    return response

def serialize(filename, data):
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    print (str(datetime.now())+" -- INFO: DATA WAS SERIALIZED SUCCESSFULLY TO FILENAME: "+filename+"\n")
            
    
def deserialize(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    print (str(datetime.now())+" -- INFO: DATA WAS DESERIALIZED SUCCESSFULLY FROM FILENAME: "+filename+"\n")
    
    return data

def generateTfidfFiles(filename, dataset):
    print (str(datetime.now())+" -- INFO: GENERATING TFIDF FILE\n")
    dataset = preprocessing(dataset)
    class_label = list(dataset['class'])
    print (str(datetime.now())+" -- INFO: TOKENIZER RUNNING\n")
    #text = [" ".join(tokenize(txt)) for txt in text]
    dataset['text'] = [" ".join(tokenize(txt)) for txt in dataset['text']]
    serialize('processed-dataset', dataset)
    print (str(datetime.now())+" -- INFO: CALCULATING TF-IDF\n")
    text = dataset['text']
    dataset_tfidf = calculateTfidf(text).toarray()
    dataset_tfidf = np.c_[dataset_tfidf, class_label]
    serialize(filename, dataset_tfidf)
    
def generateStream(original_stream, stream1, stream2, instances_per_stream, drift_instances):
    original_stream = pd.DataFrame(original_data, columns=['date', 'source', 'title', 'text', 'class'])
    aux_date = []
    for row in original_stream.iterrows():
        aux_date.append(datetime.strptime(row[1]['date'].strip(), '%Y-%m-%d %H:%M:%S'))
    original_stream['date'] = aux_date

    original_stream  = original_stream.sort_values(by=['date'])
    original_stream = original_stream.values
    
    shuffle_seed = np.arange(original_stream.shape[0])
    #np.random.shuffle(shuffle_seed)
    final_stream = np.r_[stream1, stream2]
    final_stream = final_stream[shuffle_seed]
    #original_stream = original_stream[shuffle_seed]
    
    for row in final_stream[instances_per_stream: instances_per_stream+drift_instances]:
        if (row[-1] == 1.0):
            row[-1] = 0.0
        else:
            row[-1] = 1.0
            
    for row in original_stream[instances_per_stream: instances_per_stream+drift_instances]:
        if (row[-1] == 1.0):
            row[-1] = 0.0
        else:
            row[-1] = 1.0
            
    print("Drift generated between "+str(instances_per_stream)+" and "+str(instances_per_stream+drift_instances))
    print("Stream's shape: "+str(final_stream.shape))
    
    return final_stream, original_stream

def numOfTokensPerCategory(data):
    c1 = data[data['class'] == 1]
    c0 = data[data['class'] == 0]
    
    t1 = " ".join(c1['text'])
    t0 = " ".join(c0['text'])
            
    return len(tokenize_2(t1)), len(tokenize_2(t0))
    
def totalDocuments(data):
    return len(data)

def documentsPerCategory(data):
    documents_c1 = 0
    documents_c0 = 0
    
    for row in data.iterrows():
        if (row[1]['class'] == 1):
            documents_c1 += 1
        else:
            documents_c0 += 1
            
    return documents_c1, documents_c0

def getKeywords(dataset, num):
    text = tokenize_2(" ".join(dataset['text']))
    return Counter(text).most_common(num)