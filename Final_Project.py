import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
# Import necessary ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
file_path = "data/Heart Attack Prediction.csv"
df = pd.read_csv(file_path)

# Get basic info about the dataset
data_info = {
    "Number of Rows": df.shape[0],
    "Number of Columns": df.shape[1],
    "Missing Values": df.isnull().sum().sum(),
    "Column Data Types": df.dtypes.to_dict(),
}

# Display first few rows of the dataset
df_head = df.head()

# Display dataset summary
print(data_info)
print(df_head)


#################### DATA CLEANING ######################

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert numerical columns stored as objects to proper numeric types
numeric_columns = ["trestbps", "chol", "fbs", "restecg", "thalach", "slope", "ca", "thal"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Check for missing values again
missing_values = df.isnull().sum()

# Display cleaned dataset summary
df.info()
print("missing values are: ")
print(missing_values)


# Drop columns with excessive missing values (ca, thal)
df.drop(columns=["ca", "thal"], inplace=True)

# Convert 'exang' to numeric
df["exang"] = pd.to_numeric(df["exang"], errors="coerce")

# Impute missing values:
# - For continuous variables (trestbps, chol, oldpeak, thalach), use median
# - For categorical/binary variables (fbs, restecg, slope, exang), use mode

# Impute missing values
df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
df["chol"] = df["chol"].fillna(df["chol"].median())
df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
df["restecg"] = df["restecg"].fillna(df["restecg"].mode()[0])
df["thalach"] = df["thalach"].fillna(df["thalach"].median())
df["exang"] = df["exang"].fillna(df["exang"].mode()[0])

# Handle 'slope' column (drop or impute)
df["slope"] = df["slope"].fillna(df["slope"].mode()[0])  # Option 1: Impute
# df.drop(columns=["slope"], inplace=True)  # Option 2: Drop if not useful

# Verify missing values
print(df.isnull().sum())

# Check final missing values
missing_values_after = df.isnull().sum()

# Display cleaned dataset info
df.info()
print("missing_values_after: ")
print(missing_values_after)



################ EDA ANALYSIS ####################

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# Plot histograms for numerical features
df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Check class distribution of target variable 'num'
plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df, hue="num", palette="viridis", legend=False)
plt.title("Target Variable Distribution (Heart Disease)")
plt.xlabel("Heart Disease Presence (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()



#################################### Model ###########################
####### Feature Engineering ############################

# Verify missing values are handled
assert df.isnull().sum().sum() == 0, "There are still missing values!"

# Separate features and target variable
X = df.drop(columns=["num"])  # Features
y = df["num"]  # Target variable

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Display dataset shapes
print(f"X_train.shape, X_test.shape, y_train.shape, y_test.shape: {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")



################################## ML Models ###################################################
# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Train models and evaluate performance
results = []
for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predict on test set
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results.append([name, accuracy, precision, recall, f1])

# Convert results to DataFrame and display
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print(results_df)  # Print results in console



########   Define SVM model   ##############3

svm_model = SVC()

# Define hyperparameters to tune
param_grid = {
    "C": [0.1, 1, 10, 100],  # Regularization parameter
    "kernel": ["linear", "rbf", "poly", "sigmoid"],  # Kernel type
    "gamma": ["scale", "auto"]  # Kernel coefficient
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Display best parameters and accuracy
print("Best Hyperparameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)


############ Train the best SVM model with optimized parameters Kernel = Linear ####################
best_svm = SVC(C=0.1, kernel="linear", gamma="scale")
best_svm.fit(X_train, y_train)

# Extract feature importance (absolute value of coefficients)
feature_importance = np.abs(best_svm.coef_).flatten()

# Create DataFrame for visualization
feature_importance_df = pd.DataFrame({
    "Feature": df.drop(columns=["num"]).columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Display feature importance ranking
# Print feature importance table
print("Feature Importance Ranking:")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (SVM Model)")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()

########## Train Final SVM Model with Optimized Hyperparameters #####################
final_model = SVC(C=0.1, kernel="linear", gamma="scale", probability=True)
final_model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(final_model, "final_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Final Model & Scaler Saved Successfully!")
