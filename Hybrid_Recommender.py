import pandas as pd


# Data Preparing, Creating User-Movie Matrix
rating_df = pd.read_csv("../input/movielens-20m-dataset/rating.csv")
movie_df = pd.read_csv("../input/movielens-20m-dataset/movie.csv")
def create_user_movie_df():
    df = movie_df.merge(rating_df, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df
user_movie_df = create_user_movie_df()

# Randomly selects a user from the rows of the user-movie matrix
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Determining the movies watched by the user to be suggested
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() # Movies that the user watched and rated on

# Accessing the data and IDs of users watching the same movies
movies_watched_df = user_movie_df[movies_watched] # Movies the user watched and Users who watched these movies
movies_watched_df.shape  # Number of users watching the same movies as the user. 33 movies, 138493 Users
user_movie_count = movies_watched_df.T.notnull().sum() # Total number of movies that users watched together
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False) # Users who watched more than 20 movies together
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"] # IDs of users who watched more than 20 movies together

# Most similar to the user to be suggested identify users.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
# Similarity of likes with the correlation method of users watching the same movies
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.60)][
    ["user_id_2", "corr"]].reset_index(drop=True) # The correlation of the movies watched by the selected user and other users is greater than .60
top_users = top_users.sort_values(by='corr', ascending=False) # Sort in descending order
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating_df[["userId", "movieId", "rating"]], how='inner')  # We combine movie ids and ratings with correlation table
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] # We subtract the user from the rating df given by the users

# Calculate Weighted Average Recommendation Score and keep the first 5 movies.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
# We are multiplying each other because we have a measurement problem between correlation and rating.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
# To get rid of multiplexed movieids, we average the weithed rating with groupby

recommendation_df = recommendation_df.reset_index()
recommendation_df.weighted_rating.describe().T
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.25].sort_values("weighted_rating", ascending=False)[0:5]
# We remove those whose correlation is below 3.25


######
# Suggestions with User-Based
######
movies_to_be_recommend.merge(movie_df[["movieId", "title"]])["title"]


# The id of the movie with the most recent score among the movies for which the user to suggest 5 points
movie_id = rating_df[(rating_df["userId"] == random_user) & (rating_df["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_name = movie_df[movie_df["movieId"] == movie_id]["title"].values[0]

user_ratings_movie= user_movie_df[movie_name]


user_movie_df.corrwith(user_ratings_movie).sort_values(ascending=False)[1:6].index
