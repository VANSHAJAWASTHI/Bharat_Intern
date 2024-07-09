import pandas as pd

def load_data():
    ratings = pd.read_csv('data/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None,
                         names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                                'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    data = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    user_item_matrix.fillna(0, inplace=True)
    return user_item_matrix
