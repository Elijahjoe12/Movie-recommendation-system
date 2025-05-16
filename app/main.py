# app.py
import streamlit as st
from recommend import movies, recommend_movies

# Set custom Streamlit page config
st.set_page_config(
    page_title="Movie Recommender 🎬",
    page_icon="🍿",  # You can also use a path to a .ico or .png file
    layout="centered"
)


st.title("🍿 Movie Recommendation app")

movies_list = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("🎬 Select a movie:", movies_list)

if st.button("🔃 Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)
        if recommendations is None:
            st.warning("Sorry, movie not in database.")
        else:
            st.success("You may also like:")
            st.table(recommendations)