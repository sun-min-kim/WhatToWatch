from recommendation_engine import *

file_path = '../dataset/ml-100k/u.item'
data = pd.read_csv(file_path, sep='|', usecols=[0, 1, 2], names=['movie_id', 'title', 'release_date'], header=None, encoding='latin1')
print(data.head())

user_id = 999
movies = [1, 2, 3, 4, 5]
ratings = [5, 1, 1, 1, 1]
mapped_ratings = []

for i in range(len(movies)):
    mapped_ratings.append((user_id, movies[i], ratings[i]))

recommendations = generate_recommendations(999, data, mapped_ratings)

print(recommendations)

for movie in movies:
    print(data.iat[movie, 1])

for recommendation in recommendations:
    print(data.iat[recommendation[0], 1])