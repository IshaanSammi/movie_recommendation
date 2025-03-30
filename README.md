# Movie Recommender System

## Overview
This project is a **Movie Recommender System** built using **Streamlit**, **pandas**, **NLTK**, and **scikit-learn**. It suggests similar movies based on content similarity using **Natural Language Processing (NLP)** and **cosine similarity**.

## Features
- Users can select a movie from a dropdown list.
- The system recommends 5 similar movies based on tags extracted from genres, keywords, cast, and crew.
- Uses **CountVectorizer** to convert text data into numerical vectors.
- Utilizes **cosine similarity** to find similar movies.
- Interactive web interface powered by **Streamlit**.

## Installation & Setup
### 1. Install Dependencies
Make sure you have **Python** installed. Then install the required libraries using:
```bash
pip install streamlit pandas numpy scikit-learn nltk
```

### 2. Download Dataset
Ensure you have the following datasets in CSV format:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place them in the same directory as your script.

### 3. Run the Application
Run the following command to start the Streamlit app:
```bash
streamlit run movie_recommendation.py
```
This will launch a web browser where you can select a movie and get recommendations.

## How It Works
1. The script reads and merges movie and credits datasets.
2. It extracts important features like **genres, keywords, cast, crew, and overview**.
3. Converts this data into a **tag-based textual format**.
4. Uses **stemming** to reduce words to their root form.
5. Converts tags into vectors using **CountVectorizer**.
6. Computes **cosine similarity** between movie vectors to find the most similar ones.

## Technologies Used
- **Python** (Data Processing & Machine Learning)
- **pandas, numpy** (Data Manipulation)
- **NLTK (Porter Stemmer)** (Text Preprocessing)
- **scikit-learn** (Vectorization & Similarity Computation)
- **Streamlit** (Web Interface)

## Future Improvements
- Add movie posters for better visualization.
- Implement a collaborative filtering approach.
- Improve the recommendation logic with deep learning techniques.



