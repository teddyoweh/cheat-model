import pandas as pd
import pickle
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
# Save the model to a file
 
def load_data(cheat_file, clean_file):
  sentences = []
  targets = []

 
  with open(cheat_file, 'r') as f:
    for line in f:
      sentences.append(line.strip())
      targets.append(1)

 
  with open(clean_file, 'r') as f:
    for line in f:
      sentences.append(line.strip())
      targets.append(0)

 
  df = pd.DataFrame({'text': sentences, 'target': targets})
  return df
def analyze(df_train):
    print('Word Count')
    df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
    print('Cheat Sentences: ',df_train[df_train['target']==1]['word_count'].mean())  
    print('Clean Sentences: ',df_train[df_train['target']==0]['word_count'].mean())  
 
    print('\nCharacter Count')
    df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
    print('Cheat Sentences: ',df_train[df_train['target']==1]['char_count'].mean())  
    print('Clean Sentences: ',df_train[df_train['target']==0]['char_count'].mean())  
    

    print('\nUnique Word Count')
    df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
    print('Cheat Sentences: ',df_train[df_train['target']==1]['unique_word_count'].mean())  
    print('Clean Sentences: ',df_train[df_train['target']==0]['unique_word_count'].mean())  

def save_model(model):
  with open('cheat_model.pkl', 'wb') as f:
    pickle.dump(model, f)

def load_model():
  with open('cheat_model.pkl', 'rb') as f:
    return pickle.load(f)
def preprocess(text):
    text = text.lower() 
    text=text.strip() 
    text=re.compile('<.*?>').sub('', text)  
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)   
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)  
    text = re.sub(r'\s+',' ',text)  
    
    return text
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

 
 
    
snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)
 

 
wl = WordNetLemmatizer()
 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) 
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]  
    return " ".join(a)

 
 
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

tfidf_vectorizer = load_vectorizer()
def predict_sentence(cheat_model,sentence):
    sentence = finalpreprocess(sentence)  
    X_vector = tfidf_vectorizer.transform([sentence])    
    y_predict = cheat_model.predict(X_vector)   
    y_prob = cheat_model.predict_proba(X_vector)[:, 1]   
    return y_predict, y_prob
