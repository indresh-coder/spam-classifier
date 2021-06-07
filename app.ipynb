{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template, url_for, request"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import joblib\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\Admin\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "nltk.download('stopwords')\n",
    "stop=stopwords.words(\"english\")\n",
    "\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from nltk.stem import SnowballStemmer\n",
    "ss = SnowballStemmer(\"english\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier_file_name='pickle.pkl'\n",
    "d=open(classifier_file_name,'rb')\n",
    "d\n",
    "clf= joblib.load(d)\n",
    "conversion_file_name= 'transform.pkl'\n",
    "cv= joblib.load(open(conversion_file_name,'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "txt='Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 200'\n",
    "x=cv.transform([txt]).toarray()\n",
    "import pandas as pd\n",
    "df=pd.DataFrame(x, columns=cv.get_feature_names())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "free entri week comp win fa cup final tkts st may\n"
     ]
    }
   ],
   "source": [
    "txt= clean_text(txt)\n",
    "print(txt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['len']=len(txt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>aa</th>\n",
       "      <th>aah</th>\n",
       "      <th>aaniy</th>\n",
       "      <th>aaooooright</th>\n",
       "      <th>aathi</th>\n",
       "      <th>ab</th>\n",
       "      <th>abbey</th>\n",
       "      <th>abdomen</th>\n",
       "      <th>abeg</th>\n",
       "      <th>abel</th>\n",
       "      <th>...</th>\n",
       "      <th>zf</th>\n",
       "      <th>zhong</th>\n",
       "      <th>zindgi</th>\n",
       "      <th>zoe</th>\n",
       "      <th>zogtorius</th>\n",
       "      <th>zoom</th>\n",
       "      <th>zouk</th>\n",
       "      <th>zs</th>\n",
       "      <th>zyada</th>\n",
       "      <th>len</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 6293 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   aa  aah  aaniy  aaooooright  aathi  ab  abbey  abdomen  abeg  abel  ...  \\\n",
       "0   0    0      0            0      0   0      0        0     0     0  ...   \n",
       "\n",
       "   zf  zhong  zindgi  zoe  zogtorius  zoom  zouk  zs  zyada  len  \n",
       "0   0      0       0    0          0     0     0   0      0   49  \n",
       "\n",
       "[1 rows x 6293 columns]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "s=df.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6293"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "s.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "s\n",
    "my_prediction= clf.predict(s)\n",
    "print(my_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "app= Flask(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('home.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/home')\n",
    "def home1():\n",
    "    return render_template('home.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(message):\n",
    "    message= re.sub(r'[^a-zA-Z]',' ', message)\n",
    "    message= message.lower()\n",
    "    message= message.split()\n",
    "    words= [ss.stem(word) for word in message if word not in stop]\n",
    "    return ' '.join(words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if request.method =='POST':\n",
    "        message= request.form['message']\n",
    "#         data=[message]\n",
    "        message= clean_text(message)\n",
    "        x=cv.transform([message]).toarray()\n",
    "        import pandas as pd\n",
    "        df=pd.DataFrame(x, columns=cv.get_feature_names())\n",
    "        df['len']=len(message)\n",
    "        s=df.to_numpy()\n",
    "#         vect= cv.transform(data).toarray()\n",
    "        my_prediction= clf.predict(s)\n",
    "    return render_template('result.html', prediction= my_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: Do not use the development server in a production environment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [04/Jun/2021 19:44:42] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:44:43] \"GET /favicon.ico HTTP/1.1\" 404 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:44:59] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:45:15] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:46:06] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:46:30] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:46:56] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:47:25] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [04/Jun/2021 19:48:08] \"POST /predict HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "if __name__=='__main__':\n",
    "    app.run(use_reloader=False,debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
