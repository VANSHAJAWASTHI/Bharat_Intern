def recommend_movies(movie_name, user_item_matrix, model, n_recommendations):
    if movie_name not in user_item_matrix.columns:
        return [f"Movie '{movie_name}' not found in the dataset."]
    
    movie_vector = user_item_matrix[movie_name].values.reshape(1, -1)
    distances, indices = model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)
    
    recommendations = []
    movie_list = user_item_matrix.columns
    for i in range(1, len(distances.flatten())):
        recommended_movie_index = indices.flatten()[i]
        recommended_movie_title = movie_list[recommended_movie_index]
        recommendations.append(recommended_movie_title)
    
    return recommendations
