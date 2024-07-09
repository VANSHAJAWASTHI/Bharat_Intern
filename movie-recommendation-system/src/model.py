from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def train_model(user_item_matrix):
    # Convert the user-item matrix to a sparse matrix
    sparse_matrix = csr_matrix(user_item_matrix.values)
    
    # Create and train the NearestNeighbors model
    model_knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')  # Set n_neighbors to 6
    model_knn.fit(sparse_matrix)
    
    return model_knn
# In model.py
def train_model(user_item_matrix):
    sparse_matrix = csr_matrix(user_item_matrix.values)
    print(f"Training data shape: {sparse_matrix.shape}")  # Debugging line
    model_knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    model_knn.fit(sparse_matrix)
    return model_knn

