# Movie Recommendation System

This is a simple movie recommendation system using collaborative filtering.

## Project Structure

- `data/`: Contains the datasets (movies and ratings).
- `src/`: Source code for data loading, model training, and making recommendations.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies required for the project.
- `run.py`: Script to run the recommendation system.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the MovieLens 100k datasets (`u.data` and `u.item`) in the `data/` directory.

2. Run the recommendation system:
    ```bash
    python run.py
    ```

## License

This project is licensed under the MIT License.
