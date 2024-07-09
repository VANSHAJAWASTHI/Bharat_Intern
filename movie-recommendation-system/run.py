from src.data_loader import load_data
from src.model import train_model
from src.recommend import recommend_movies

def main():
    user_item_matrix = load_data()
    model_knn = train_model(user_item_matrix)

    movie_name = 'Toy Story (1995)'
    if movie_name not in user_item_matrix.columns:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return

    recommendations = recommend_movies(movie_name, user_item_matrix, model_knn, 5)
    print(f'Recommendations for {movie_name}:')
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()
