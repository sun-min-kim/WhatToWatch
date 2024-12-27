from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import pandas as pd


# Load MovieLens 100K dataset and prepare for scikit-surprise
file_path = '../dataset/ml-100k/u.data'
data = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

trainset, testset = train_test_split(surprise_data, test_size=0.2)

# Use SVD algorithm for recommendation
model = SVD()
model.fit(trainset)


# Function to get movie recommendations
def get_movie_recommendations(user_id, n_recommendations=5):

    all_movie_ids = data['item_id'].unique()

    # List of (movie_id, estimated_rating) tuples
    estimated_ratings = []
    for movie_id in all_movie_ids:
        est_rating = model.predict(user_id, movie_id).est
        estimated_ratings.append((movie_id, est_rating))

    # Sort movies based on estimated ratings and get the top N
    estimated_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = estimated_ratings[:n_recommendations]

    return top_n_movies


# Function to be used to retrieve recommendations externally
def generate_recommendations(user_id, data, new_user_ratings):

    # Add new ratings to dataset and prepare dataset
    new_data = pd.DataFrame(new_user_ratings, columns=['user_id', 'item_id', 'rating'])
    updated_data = pd.concat([data, new_data], ignore_index=True)

    surprise_data = Dataset.load_from_df(updated_data[['user_id', 'item_id', 'rating']], reader)

    # Train model again with updated dataset
    trainset = surprise_data.build_full_trainset()
    model.fit(trainset)

    # Get movie recommendations for new user
    recommendations = get_movie_recommendations(user_id)

    return recommendations
