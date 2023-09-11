#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load and preprocess data
feature_file = pd.read_csv('/Users/milazavadjil/Downloads/12sl_features.csv')
feature_file = feature_file.iloc[:1500]
feature_matrix = feature_file.to_numpy()
imputer = SimpleImputer(missing_values = np.nan,
                        strategy ='mean')
imputer = imputer.fit(feature_matrix)  
feature_matrix = imputer.transform(feature_matrix)
print(feature_matrix.shape)

label_file = pd.read_excel(r'/Users/milazavadjil/Downloads/ECG Data Labels.xlsx')
label_table = pd.DataFrame(label_file, columns=['Label'])
label_table = label_table.iloc[:1500]
label_matrix = label_table.to_numpy()
print(label_matrix.shape)

# Train-test split
X = feature_matrix
y = label_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply feature selection using SelectKBest
k_best = SelectKBest(score_func=f_classif, k=10)  # You can change 'k' to select a different number of top features
X_train_k_best = k_best.fit_transform(X_train_scaled, y_train)
X_val_k_best = k_best.transform(X_val_scaled)
X_test_k_best = k_best.transform(X_test_scaled)

# Train model
model = DecisionTreeClassifier()

# Define the hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'splitter': ['best', 'random'],  # Splitting strategy
    'class_weight': [None, 'balanced', {0: 1, 1: 2}]  # Weight for classes
}

# Create the GridSearchCV object
grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1  # Use all available CPU cores
)

# Perform the grid search on the training data
grid_search.fit(X_train_k_best, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Validation set evaluation with the best model
val_predictions = best_model.predict(X_val_k_best)
accuracy = accuracy_score(y_val, val_predictions)
precision = precision_score(y_val, val_predictions, average='weighted')
recall = recall_score(y_val, val_predictions, average='weighted')
f1 = f1_score(y_val, val_predictions, average='weighted')
val_classification_report = classification_report(y_val, val_predictions)

# Test set evaluation with the best model
test_predictions = best_model.predict(X_test_k_best)
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, average='weighted')
recall = recall_score(y_test, test_predictions, average='weighted')
f1 = f1_score(y_test, test_predictions, average='weighted')
test_classification_report = classification_report(y_test, test_predictions)

# Print the best hyperparameters and evaluation metrics
print("Best hyperparameters:", grid_search.best_params_)
print("Test Set Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Validation Set Classification Report:")
print(val_classification_report)
print("Test Set Classification Report:")
print(test_classification_report)

# Confusion matrix
label_names = ['$SWQR', 'ACCEL', 'AFIB', 'EAR', 'FLUT', 'NSR', 'QCERR', 'SBRAD', 'SRTH', 'STACH', 'VPR']
cm = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Feature importance visualization
feature_df = pd.DataFrame(X_train_k_best)
selected_feature_names = feature_df.columns
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(range(X_train_k_best.shape[1]), feature_importances, align='center')
plt.yticks(range(X_train_k_best.shape[1]), selected_feature_names)  # Use actual feature names
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
