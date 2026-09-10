# 1. Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 2. Load the dataset

df = pd.read_csv("car_evaluation.csv")

print(df.head())


# 3. Give names to the columns

df.columns = [
    "buying",
    "maintenance",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]


# 4. Convert text values into numbers

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])


# 5. Separate input and output

X = df.drop("class", axis=1)
y = df["class"]


# 6. Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Create the ML models

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42)
}


# 8. Train and evaluate each model

results = []

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])


# 9. Compare the results

results_df = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nPerformance Comparison:")
print(results_df)


# 10. Find the best algorithm

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Performing Algorithm:")
print(best_model["Algorithm"])
print("Accuracy:", best_model["Accuracy"])
