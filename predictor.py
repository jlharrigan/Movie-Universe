#!/usr/bin/env python
# coding: utf-8

# In[4]:

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.preprocessing import MultiLabelBinarizer
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import time
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import string
from concurrent.futures import ThreadPoolExecutor, wait
from random import uniform
import random
import pickle
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.preprocessing import MultiLabelBinarizer
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
from sklearn.model_selection import train_test_split
import dill

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras import regularizers
from random import sample
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# In[5]:


with open('wd_popularity.pkl', 'rb') as f:
    wd_pop = pickle.load(f)
wd_pop_names = list(wd_pop.keys())
    
with open('actor_popularity.pkl', 'rb') as f:
    actor_pop = pickle.load(f)
cast_pop_names = list(actor_pop.keys())

all_dict = wd_pop | actor_pop
for key, val in list(all_dict.items()):
    all_dict[key]['average'] = round(np.mean(list(val.values())))
    for key1, val1 in list(val.items()):
        if val1 >= 1000000:
            all_dict[key][key1] = 1000000


# In[68]:


def pred_dim_reduction(baseline_df, column, genre):
    with open('./' + genre + '/vect_act.pkl', 'rb') as f:
        vect_act = dill.load(f)
    with open('./' + genre + '/vect_dir.pkl', 'rb') as f:
        vect_dir = dill.load(f)
    with open('./' + genre + '/vect_write.pkl', 'rb') as f:
        vect_write = dill.load(f)
    with open('./' + genre + '/vect_key.pkl', 'rb') as f:
        vect_key = dill.load(f)
    with open('./' + genre + '/vect_prod.pkl', 'rb') as f:
        vect_prod = dill.load(f)
    with open('./' + genre + '/actor_columns.pkl', 'rb') as f:
        ACT_DATA = pickle.load(f)
    with open('./' + genre + '/director_columns.pkl', 'rb') as f:
        DIR_DATA = pickle.load(f)
    with open('./' + genre + '/writer_columns.pkl', 'rb') as f:
        WRITE_DATA = pickle.load(f)
    with open('./' + genre + '/key_columns.pkl', 'rb') as f:
        KEY_DATA = pickle.load(f)
    with open('./' + genre + '/prod_columns.pkl', 'rb') as f:
        PROD_DATA = pickle.load(f)
    
    if column == 'actors':
        vectorizer = vect_act
        good_data = ACT_DATA
    elif column == 'director':
        vectorizer = vect_dir
        good_data = DIR_DATA
    elif column == 'writer':
        vectorizer = vect_write
        good_data = WRITE_DATA
    elif column == 'key_words':
        vectorizer = vect_key
        good_data = KEY_DATA
    elif column == 'prod_comp':
        vectorizer = vect_prod
        good_data = PROD_DATA
    
    count_data = vectorizer.transform(baseline_df[column])   
    movie_data = pd.DataFrame(data = count_data.toarray(), index = baseline_df.index, columns = vectorizer.get_feature_names())
    movie_data.columns = [name + ' ' + column for name in list(movie_data.columns)]
    
    return baseline_df.join(movie_data[good_data], lsuffix = '_left').drop(column, axis = 1)

# In[69]:

def pop(df, pred_names):
    for name in pred_names:
        try:
            original_name = name.replace(' actors', '').replace(' director', '').replace(' writer' , '')
            pop_scores = all_dict[original_name]
            
        except:
            pop_scores = {'average': 0}
        
        year = df.release_date[0]
        
        try:
            score = (1000000 - pop_scores[year])
        except:
            score = (1000000 - pop_scores['average'])
        
        if ' actors' in name:
            df.cast_pop = df.cast_pop + score
        elif ' director' in name: 
            df.dir_pop = df.dir_pop + score
        elif ' writer' in name: 
            df.write_pop = df.write_pop + score
    return df

# In[70]:

def make_pred_dataset(df, genre):
    with open('./' + genre + '/mlb_genres.pkl', 'rb') as f:
        mlb_genres = pickle.load(f)
    with open('./' + genre + '/mlb_languages.pkl', 'rb') as f:
        mlb_languages = pickle.load(f)
    with open('./' + genre + '/mlb_country.pkl', 'rb') as f:
        mlb_country = pickle.load(f)
    with open('./' + genre + '/ohe.pkl', 'rb') as f:
        ohe = pickle.load(f)
    
    tdf = deepcopy(df)
    tdf = tdf.join(pd.DataFrame.sparse.from_spmatrix(mlb_genres.transform(tdf.pop('genres')), index = tdf.index, columns = mlb_genres.classes_))
    tdf = tdf.reset_index().drop('index', axis =1)
    tdf = tdf.drop(['Biography', 'Documentary', 'Film-Noir', 'Music', 'Musical', 'News', 'Reality-TV', 'Sport'], axis = 1)
    tdf['dir_pop'] = 0
    tdf['write_pop'] = 0
    tdf['cast_pop'] = 0      
    tdf = tdf.join(pd.DataFrame.sparse.from_spmatrix(mlb_languages.transform(tdf.pop('languages')), index = tdf.index, columns = mlb_languages.classes_), lsuffix = '_left')
    tdf = tdf.join(pd.DataFrame.sparse.from_spmatrix(mlb_country.transform(tdf.pop('country')), index = tdf.index, columns = mlb_country.classes_), lsuffix = '_left')

    tdf = pred_dim_reduction(tdf, 'prod_comp', genre)
    tdf = pred_dim_reduction(tdf, 'key_words', genre)
    temp = tdf.shape[1]
    tdf = pred_dim_reduction(tdf, 'actors', genre)
    tdf = pred_dim_reduction(tdf, 'director', genre)
    tdf = pred_dim_reduction(tdf, 'writer', genre)  
    
    rating = pd.DataFrame(ohe.transform(tdf[['m_rating']]).toarray(), columns = ohe.get_feature_names(['m_rating']))
    rating.columns = [col[9:] for col in list(rating.columns)]
    rating.index = tdf.index
    tdf = pd.concat([tdf, rating], 1).drop('m_rating', axis =1)
    
    return(tdf)


# In[71]:


user_agents = [ 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36', 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36', 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148', 
    'Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:106.0) Gecko/20100101 Firefox/106.0',
    'Mozilla/5.0 (Windows NT 10.0; rv:106.0) Gecko/20100101 Firefox/106.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:107.0) Gecko/20100101 Firefox/107.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:107.0) Gecko/20100101 Firefox/107.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0',
    'Mozilla/5.0 (Windows NT 10.0; rv:107.0) Gecko/20100101 Firefox/107.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.56',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.42',
    'Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0'
] 


# In[72]:


def get_movie(movie_url):
    time.sleep(uniform(0,2))
    user_agent = random.choice(user_agents) 
    headers = {'User-Agent': user_agent, 'Accept-Language': 'en-US,en;q=0.5'} 
    req = requests.get('https://www.imdb.com/' + movie_url, headers = headers)
    soup = BeautifulSoup(req.content, 'html.parser')
    
    if str(req) != '<Response [200]>':
        return 0

    try:
        title = soup.find('h1', class_ = 'sc-b73cd867-0 eKrKux').text
    except:
        try: 
            title = soup.find('h1', class_ = 'sc-b73cd867-0 fbOhB').text
        except:
            try:
                title = soup.find('h1', class_ = 'sc-b73cd867-0 cAMrQp').text
            except:
                title = 'none'
    try:
        wr = soup.find_all('ul', role = "presentation", class_ = 'ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')
        try:
            writers = [w.text.split('(')[0] + ' ' + w.a['href'].split('nm')[1].split('/?')[0] for w in wr[1].findAll('li')]
        except:
            writers = ''
        try:
            director = [w.text.split('(')[0] + ' ' + w.a['href'].split('nm')[1].split('/?')[0] for w in wr[0].findAll('li')]
        except:
            director = ''
    except:
        writers = ''
        director = ''
    
    try:
        avg_review = soup.find('span', class_="sc-7ab21ed2-1 jGRxWM").text
    except:
        avg_review = 20
    
    try:
        raw = soup.find('div', class_="sc-7ab21ed2-3 dPVcnq").text
        if raw[-1] == 'M':
            num_reviews = float(raw[:-1]) * 1000000
        elif raw[-1] == 'K':
            num_reviews = float(raw[:-1]) * 1000
        else:
            num_reviews = float(raw)
    except:
        num_reviews = 0

    try:
        budget = int(soup.find('button', text = 'Budget').findNext('div').text.partition('(')[0][1:].strip().replace(',', ''))
    except:
        try:
            budget = int(soup.find('button', text = 'Budget').findNext('div').text[4:].split('(')[0].split()[0].replace(',', ''))
        except:
            budget = 0
    try:
        rating = soup.find_all('a', class_ = 'ipc-link ipc-link--baseAlt ipc-link--inherit-color sc-8c396aa2-1 WIUyh')[1].text
    except:
        rating = 'UNK'
        
    try:
        date = soup.find('a', text = 'Release date').findNext('div').text.partition('(')[0].strip()
    except:
        date = '?'
        
    try:
        country = [soup.find('button', text = 'Country of origin').findNext('div').text]
    except:
        try:
            countries_raw = soup.find('button', text = 'Countries of origin').next_sibling()[0].findAll('li')
            country = [c.text for c in countries_raw]
        except:
            country = ['unknown']
    
    genres = []
    try:
        gs = soup.find_all('a', class_ ='sc-16ede01-3 bYNgQ ipc-chip ipc-chip--on-baseAlt')
        for g in gs:
            genres.append(g.text)
    except:
        genres = ['none']
    
    all_comps = []
    try:
        comps = soup.find('a', text = 'Production companies').findNext('div').find_all('li')
        for comp in comps:
            all_comps.append(comp.text)
    except:
        all_comps = ['none']
    
    all_cast = []
    try:
        try:
            cast = soup.findAll('a', class_ = 'sc-bfec09a1-1 gfeYgX')[:10]
        except:
            cast = soup.findAll('a', class_ = 'sc-bfec09a1-1 gfeYgX')
            
        for act in cast:
            try:
                all_cast.append(act.text + ' ' + act['href'].split('nm')[1].split('/?')[0])
            except:
                all_cast.append('no info')
    except:
        all_cast = ['none']
    
    try:
        runtime = soup.find('button', text = 'Runtime').findNext('div').text.split()
        run = int(runtime[0]) * 60
        if len(runtime) > 2:
            run = run + int(runtime[2])
    except:
        run = 0
    
    all_langs = []
    try:
        langs = soup.find('button', text = 'Languages').findNext('div').ul.find_all('li')
        for lang in langs:
            all_langs.append(lang.text)
    except:
        try:
            all_langs.append(soup.find('button', text = 'Language').findNext('div').ul.li.text)
        except:
            all_langs = ['none']
    
    return {'id' : movie_url[9:-1], 'title': title, 'writer': writers, 'director': director, 'actors': all_cast,
                       'num_review': num_reviews,'release_date': date, 'country': country, 'imdb_rating': avg_review, 
                       'm_rating': rating, 'languages': all_langs, 'runtime': run, 'genres': genres, 'budget': budget,
                       'prod_comp': all_comps}, gs[0].text


# In[73]:


def get_keys(movie_url):
    user_agent = random.choice(user_agents) 
    headers = {'User-Agent': user_agent, 'Accept-Language': 'en-US,en;q=0.5'} 
  
    req = requests.get('https://www.imdb.com/' + movie_url + 'keywords', headers = headers)
    soup = BeautifulSoup(req.content, 'html.parser')

    if str(req) != '<Response [200]>':
        return 0

    all_words = []
    try:
        words = soup.find_all('div', class_ = 'sodatext')
        
        for word in words:
            all_words.append(word.text.strip())
    except:
           pass
    
    return {'id': movie_url[9:-1], 'key_words': all_words}


# In[74]:

def make_movie(url):
    m_df = pd.DataFrame(columns = ['id', 'title', 'release_date', 'runtime', 'm_rating', 'imdb_rating',
       'num_review', 'director', 'writer', 'actors', 'genres', 'country',
       'languages', 'prod_comp'])
    m, g = get_movie(url)
    m_df = m_df.append(m, ignore_index = True)
    
    k_df = pd.DataFrame(columns = ['id', 'key_words'])
    k = get_keys(url)
    k_df = k_df.append(k, ignore_index = True)
    
    films_df = pd.merge(m_df, k_df, how = 'outer', on = 'id')
    films_df = films_df.dropna(subset = 'title')
    films_df = films_df.drop('budget', axis = 1)
    return films_df, g

# In[75]:

def search_for_movie(title):
    title = title.replace(' ', '%20').replace(':', '3A')
    
    user_agent = random.choice(user_agents) 
    headers = {'User-Agent': user_agent, 'Accept-Language': 'en-US,en;q=0.5'} 
  
    req = requests.get('https://www.imdb.com/find/?q=' + title + '&ref_=nv_sr_sm', headers = headers)
    soup = BeautifulSoup(req.content, 'html.parser')
    
    table = soup.find('ul', class_ = 'ipc-metadata-list ipc-metadata-list--dividers-after sc-17bafbdb-3 aQtNB ipc-metadata-list--base')
    movies = table.find_all('div', class_ = 'ipc-metadata-list-summary-item__tc')
    
    films = []
    for movie in movies:
        title = movie.a.text
        link = movie.a['href']
        date = movie.find('label', class_ = 'ipc-metadata-list-summary-item__li').text
        films.append((title, date, link))
    return films

# In[76]:


# In[77]:
def make_prediction(df, genre):
    new_df = make_pred_dataset(df, gen)
    pred_names = [w + ' ' + 'writer' for w in df.writer[0]] + [w + ' ' + 'actors' for w in df.actors[0]] + [w + ' ' + 'director' for w in df.director[0]]
    
    new_final = pop(new_df, pred_names)
    new_final = new_final.drop(['id', 'title', 'imdb_rating'], axis = 1)
    new_final.release_date = pd.to_datetime(new_final.release_date)
    new_final.release_date = new_final.release_date.apply(lambda x: x.toordinal())
    with open('./' + genre + '/ss.pkl', 'rb') as f:
        ss = pickle.load(f)
    pred_scaled = pd.DataFrame(ss.transform(new_final),columns = new_final.columns)
    
    with open('./' + genre + '/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    pca_pred = pca.transform(pred_scaled)
    
    model = load_model('./'+ genre + '/' + genre.lower() + '_model.h5')
    
    return model.predict(pca_pred)
    
    

# In[67]:

st.title('The Movie Predictor')



# In[78]:

title = st.text_input('Movie title', 'Die Hard')
st.write('possible movies')
movies = search_for_movie(title)
for movie in movies:
    if st.button(movie[0] + ' ' + movie[1]):
        df, gen = make_movie(movie[2])
        st.write(make_prediction(df, gen)[0][0])


    


# In[79]:



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




