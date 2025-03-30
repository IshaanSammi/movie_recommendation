import streamlit as st
import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("C:/Users/HP/Downloads/tmdb_5000_movies.csv/tmdb_5000_movies.csv")
credits = pd.read_csv("C:/Users/HP/Downloads/tmdb_5000_credits.csv/tmdb_5000_credits.csv")


movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    l = []
    for i in ast.literal_eval(obj):
        if len(l) < 3:
            l.append(i['name'])
        else:
            break
    return l

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())


ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [new_df.iloc[i[0]].title for i in movie_list]
    except:
        return []


st.title("Movie Recommender System")
selected_movie = st.selectbox("Choose a movie", new_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
    else:
        st.write("No recommendations found.")
