import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2, 3])
    ])

classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

unique_classes = np.unique(y)
for cls in unique_classes:
    print(f"Class {cls} - {iris.target_names[cls]}:")
    tp = confusion_matrix(y_test, y_pred)[cls, cls]
    fp = np.sum(confusion_matrix(y_test, y_pred)[:, cls]) - tp
    fn = np.sum(confusion_matrix(y_test, y_pred)[cls, :]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-score: {f1_score:.2f}")
    print()

