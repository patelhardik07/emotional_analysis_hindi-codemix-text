import pandas as pd
import numpy as np
import itertools as it
import spacy
from spacy.lang.hi import Hindi
import regex as re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
nlp_hi = Hindi()


    
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
    
    # get data
  data = request.get_json(force=True)
  sent = data['comment']
  res={}
  af_pre=preprocessing_hi(sent)
  print(af_pre)
  res['after']=af_pre
  res['before']=sent
                
  return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
