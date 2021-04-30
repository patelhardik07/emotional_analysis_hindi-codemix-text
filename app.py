import pandas as pd
import numpy as np
import itertools as it
import spacy
from spacy.lang.hi import Hindi
import regex as re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
nlp_hi = Hindi()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

loaded_model = tf.keras.models.load_model('LSTM_hindi.h5')    
def preprocessing_hi(text_hi):
  tweet_hi = []
  tokenized_text = nlp_hi(text_hi)
  for token in tokenized_text:
    if(token.text!='\n\n' 
        and not token.is_stop 
        and not token.is_punct 
        and not token.is_space 
        and not token.like_email
        and not token.is_digit
        and not token.is_quote
        and not token.is_alpha
        and not token.like_url):
        tweet_hi.append(token)
  tweet = ' '.join([str(token)  for token in tweet_hi])
   
  return tweet
labels = ['Negative-Anger', 'Negative-fear', 'Negative-sadness', 'Nuetral', 'Positive-Joy', 'Positive-Surprise', 'Positive-Trust', 'Satire']
dat = pd.read_csv('dataset.csv', sep=',' ,names=["message", "sentiment"])
dat=dat.drop(index=0)
dat=dat.dropna()
dat=dat.drop(dat[dat['sentiment'] == "Discard"].index) 
dat.reset_index(inplace = True, drop = True) 
corpus=[]
for i in range(0, len(dat)):
  comment = dat['message'][i]
  comment_af = preprocessing_hi(comment)
  corpus.append(comment_af)
dat['new_messages'] = corpus

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(dat['new_messages'].values)

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})
# routes
    
@app.route('/', methods=['POST'])
#@crossdomain(origin='*')
def predict():
  data = request.get_json(force=True)
  res={}
   # get data
  for i in (data['comment']):
    sent = data['comment'][i]
    cmt=sent
    af_pre=preprocessing_hi(sent)
    seq = tokenizer.texts_to_sequences([af_pre])
    padded = pad_sequences(seq,maxlen=137)
    predictions = loaded_model.predict(padded)
    pr=np.argmax(predictions)
    res[i]={}
    res[i]['Comment']=cmt
    res[i]['Emotion']= labels[pr]           
  return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
