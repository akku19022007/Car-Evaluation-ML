from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# ============================================================
# CAR EVALUATION USING MACHINE LEARNING
# ============================================================


# 1. IMPORT LIBRARIES


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("car_evaluation.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. GIVE NAMES TO THE COLUMNS
# ============================================================

df.columns = [
    "buying",
    "maintenance",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 5. CONVERT TEXT VALUES INTO NUMBERS
# ============================================================

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("class", axis=1)
y = df["class"]


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 8. K-NEAREST NEIGHBORS
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(
    y_test, y_pred_knn, average="weighted"
)
knn_recall = recall_score(
    y_test, y_pred_knn, average="weighted"
)
knn_f1 = f1_score(
    y_test, y_pred_knn, average="weighted"
)


# ============================================================
# 9. DECISION TREE
# ============================================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(
    y_test, y_pred_dt, average="weighted"
)
dt_recall = recall_score(
    y_test, y_pred_dt, average="weighted"
)
dt_f1 = f1_score(
    y_test, y_pred_dt, average="weighted"
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(
    y_test, y_pred_rf, average="weighted"
)
rf_recall = recall_score(
    y_test, y_pred_rf, average="weighted"
)
rf_f1 = f1_score(
    y_test, y_pred_rf, average="weighted"
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        knn_accuracy,
        dt_accuracy,
        rf_accuracy
    ],

    "Precision": [
        knn_precision,
        dt_precision,
        rf_precision
    ],

    "Recall": [
        knn_recall,
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        knn_f1,
        dt_f1,
        rf_f1
    ]
})


print("\n================ MODEL COMPARISON ================")

print(results)


# ============================================================
# 12. ACCURACY COMPARISON GRAPH
# ============================================================

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Accuracy Comparison of ML Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.show()


# ============================================================
# 13. ALL METRICS COMPARISON GRAPH
# ============================================================

results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Performance Comparison of ML Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.show()


# ============================================================
# 14. FIND BEST MODEL
# ============================================================

best_model = results.loc[
    results["Accuracy"].idxmax()
]


print("\n================ BEST PERFORMING ALGORITHM ================")

print("Best Algorithm :", best_model["Model"])
print("Accuracy       :", best_model["Accuracy"])
print("Precision      :", best_model["Precision"])
print("Recall         :", best_model["Recall"])
print("F1 Score       :", best_model["F1 Score"])


# ============================================================
# 15. CONCLUSION
# ============================================================

print("\n================ CONCLUSION ================")

print(
    "Three machine learning algorithms were applied "
    "to the Car Evaluation dataset."
)

print(
    "The algorithms were compared using Accuracy, "
    "Precision, Recall and F1 Score."
)

print(
    "The best-performing algorithm is:",
    best_model["Model"]
)
