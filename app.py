import streamlit as st
from recommender import MovieRecommender


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


# Load recommendation system
@st.cache_resource
def load_recommender():

    return MovieRecommender()


recommender = load_recommender()


# Title
st.title("🎬 Movie Recommendation System")

st.write(
    "Get movie recommendations based on your favorite movie "
    "or preferred genres."
)


st.divider()


# -------------------------------
# OPTION 1: Movie-based recommendation
# -------------------------------

st.header("🔎 Recommend Movies Based on a Movie")

movie_list = sorted(
    recommender.movies["title"].tolist()
)

selected_movie = st.selectbox(
    "Select a movie you like:",
    movie_list
)


if st.button("Recommend Similar Movies"):

    recommendations = recommender.recommend_by_movie(
        selected_movie,
        top_n=5
    )

    st.subheader("🎥 Top 5 Recommendations")

    if recommendations.empty:

        st.warning("No recommendations found.")

    else:

        for index, row in recommendations.iterrows():

            st.write(
                f"**{index + 1}. {row['title']}**"
            )

            st.caption(
                f"Genres: {row['genres']}"
            )


st.divider()


# -------------------------------
# OPTION 2: Genre-based recommendation
# -------------------------------

st.header("🎯 Recommend Movies by Genre")

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]


selected_genres = st.multiselect(
    "Choose your preferred genres:",
    genres
)


if st.button("Recommend by Genre"):

    if not selected_genres:

        st.warning(
            "Please select at least one genre."
        )

    else:

        recommendations = (
            recommender.recommend_by_genre(
                selected_genres,
                top_n=5
            )
        )

        st.subheader("🎬 Recommended Movies")

        if recommendations.empty:

            st.warning(
                "No movies found for the selected genres."
            )

        else:

            for index, row in recommendations.iterrows():

                st.write(
                    f"**{index + 1}. {row['title']}**"
                )

                st.caption(
                    f"Genres: {row['genres']}"
                )


st.divider()

st.info(
    "Recommendation technique: Content-Based Filtering "
    "using TF-IDF and Cosine Similarity."
)