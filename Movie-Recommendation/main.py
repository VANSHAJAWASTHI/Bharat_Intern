import pandas as pd
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise.accuracy import rmse, mae
from surprise.prediction_algorithms.random_pred import NormalPredictor

data = Dataset.load_builtin('ml-100k')

algo = SVD()

cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("Mean RMSE:", cv_results['test_rmse'].mean())
print("Mean MAE:", cv_results['test_mae'].mean())

trainset, testset = train_test_split(data, test_size=0.25)

algo.fit(trainset)

predictions = algo.test(testset)

print(f"\nRMSE on test set: {rmse(predictions)}")
print(f"MAE on test set: {mae(predictions)}")

user_id = str(196)
item_id = str(302)
pred = algo.predict(user_id, item_id, verbose=True)

fallback_algo = NormalPredictor()

cv_results_fallback = cross_validate(fallback_algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("Mean RMSE (Fallback):", cv_results_fallback['test_rmse'].mean())
print("Mean MAE (Fallback):", cv_results_fallback['test_mae'].mean())
