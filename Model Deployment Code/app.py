import streamlit as st

import requests
import html5lib
from bs4 import BeautifulSoup

import pandas as pd

from ipynb.fs.full.preprocessing_functions import lower_text, initial_clean, fix_contractions, final_clean, remove_extra_space, remove_stopwords, lemmatize_review, convert_to_tokens
from ipynb.fs.full.preprocessing_functions import get_movie_details, get_poster, get_url, get_critics_review, get_audience_review, sentiment_analysis, recommend

import warnings
warnings.filterwarnings('ignore')

#Code Starts here
def color_results(x):
    if x == 'Positive':
        color = '#00ab41'
    else:
        color = '#df2c14'
    return f"background-color: {color}"

#Importing Dataframes
metadata = pd.read_csv('clean_metadata.csv')
tag_data = pd.read_csv('final_clean.csv')
scaled_transformed = convert_to_tokens()

#Creating the page
st.set_page_config(page_title = 'Movie Recommendation System',layout='wide')
st.title('Movie Recommendation System',)

movie_title = st.selectbox('Select Movie',options = metadata['title'].unique(),index = None)

if movie_title:
    col1,col2 = st.columns([0.2,0.8])

    with col1:
        st.image(get_poster(movie_title))

    with col2:
        overview,genre,director,film_cast = get_movie_details(movie_title)
        st.write(overview)
        st.write('Directed by : ',director)
        st.write('Genre : ',genre)
        st.write('Cast : ',film_cast)
    

    if st.button('Recommend'):
        with st.spinner('Finding movies you might like'):
            recommendations = recommend(movie_title)
            critics_sentiments = sentiment_analysis(get_critics_review(movie_title))
            audience_sentiments = sentiment_analysis(get_audience_review(movie_title))

        st.subheader('Recommendations')
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(get_poster(recommendations.iloc[i]['Title']))
                st.write(recommendations.iloc[i]['Title'])

        st.subheader('Critics Reviews')
        if critics_sentiments.shape[0] == 0:
            st.write('No critics reviews found :(')
        else:
            st.table(critics_sentiments.style.map(color_results,subset = 'Sentiment'))

        st.subheader('Audience Reviews')
        if audience_sentiments.shape[0] == 0:
            st.write('No audience reviews found :(')
        else:
            st.table(audience_sentiments.style.map(color_results,subset = 'Sentiment'))