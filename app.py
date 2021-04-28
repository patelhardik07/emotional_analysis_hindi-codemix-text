import pandas as pd
import numpy as np
import itertools as it
import spacy
from spacy.lang.hi import Hindi
import regex as re
nlp_hi = Hindi()



extended_stop_words = ['जी','श्री','|','l','श्रीमती']
for stopword in extended_stop_words:
    lexeme = nlp_hi.vocab[stopword]
    lexeme.is_stop = True
    
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
        and (re.search(r'@\S+',token.text) is None)
        and not token.like_url):
      tweet_hi.append(token.lemma_)

  
  tweet = ' '.join([token  for token in tweet_hi])
   
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
  res['after']=af_pre
                 
  return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
