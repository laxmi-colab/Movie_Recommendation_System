import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:

    def __init__(self, file_path="data/movies.csv"):

        self.movies = pd.read_csv(file_path)

        # Remove missing values
        self.movies = self.movies.dropna(
            subset=["title", "genres"]
        ).reset_index(drop=True)

        # Clean genres
        self.movies["clean_genres"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
        )

        # Convert genres into TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.genre_matrix = self.vectorizer.fit_transform(
            self.movies["clean_genres"]
        )

        # Movie title -> index
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies["title"]
        ).drop_duplicates()

    def recommend_by_movie(self, movie_title, top_n=5):

        if movie_title not in self.indices:
            return pd.DataFrame()

        movie_index = self.indices[movie_title]

        # Calculate similarity ONLY for the selected movie
        movie_vector = self.genre_matrix[movie_index]

        similarity_scores = cosine_similarity(
            movie_vector,
            self.genre_matrix
        ).flatten()

        # Get top similar movies
        similar_indices = similarity_scores.argsort()[
            -top_n - 1:
        ][::-1]

        # Remove the selected movie itself
        similar_indices = [
            i for i in similar_indices
            if i != movie_index
        ][:top_n]

        result = self.movies.iloc[
            similar_indices
        ][["title", "genres"]].copy()

        return result

    def recommend_by_genre(
        self,
        selected_genres,
        top_n=5
    ):

        selected_genres = [
            genre.lower()
            for genre in selected_genres
        ]

        scores = []

        for index, row in self.movies.iterrows():

            movie_genres = row["genres"].lower()

            score = 0

            for genre in selected_genres:

                if genre in movie_genres:
                    score += 1

            if score > 0:
                scores.append(
                    (index, score)
                )

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        selected_indices = [
            item[0]
            for item in scores[:top_n]
        ]

        result = self.movies.iloc[
            selected_indices
        ][["title", "genres"]].copy()

        return result