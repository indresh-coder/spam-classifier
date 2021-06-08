# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:15:42 2021

@author: Admin
"""

from flask import Flask, render_template, url_for, request
import pickle
import joblib
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop=stopwords.words("english")

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("english")

import os
chd=os.curdir

classifier_file_name='pickle.pkl'
d=open(classifier_file_name,'rb')
d
clf= joblib.load(d)
conversion_file_name= 'transform.pkl'
cv= joblib.load(open(conversion_file_name,'rb'))

txt='Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 200'
x=cv.transform([txt]).toarray()
import pandas as pd
df=pd.DataFrame(x, columns=cv.get_feature_names())
def clean_text(message):
    message= re.sub(r'[^a-zA-Z]',' ', message)
    message= message.lower()
    message= message.split()
    words= [ss.stem(word) for word in message if word not in stop]
    return ' '.join(words)
txt= clean_text(txt)
print(txt)
df['len']=len(txt)
s=df.to_numpy()
s
my_prediction= clf.predict(s)
print(my_prediction)
app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home1():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        message= request.form['message']
#         data=[message]
        message= clean_text(message)
        x=cv.transform([message]).toarray()
        import pandas as pd
        df=pd.DataFrame(x, columns=cv.get_feature_names())
        df['len']=len(message)
        s=df.to_numpy()
#         vect= cv.transform(data).toarray()
        my_prediction= clf.predict(s)
    return render_template('result.html', prediction= my_prediction)

if __name__=='__main__':
    app.run(use_reloader=False,debug=True)
