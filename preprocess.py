import pandas as pd


def load_movies():
    movies = pd.read_csv("data/movies.csv")

    movies = movies.dropna(subset=["title", "genres"])

    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

    return movies