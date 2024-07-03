Python code for analysis

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# Load the dataset
file_path = '/Users/user/Downloads/diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Basic statistical summary
print(data.describe(include='all'))

#Data Cleaning

any_nans = data.isnull().values.any()
print(any_nans)

#Check for NaN per column
any_nans_per_column = data.isnull().any()
print(any_nans_per_column)

#Checking for NaN in the dataset
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.show()

# Preprocessing the data
# Convert categorical data to numeric
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})

# Create a mapping for the 'smoking_history' categorical data to numeric values
smoking_history_mapping = {
    'never': 0,
    'No Info': 1,
    'current': 2,
    'former': 3,
    'ever': 4,
    'not current': 5
}


# Apply the mapping directly to the 'smoking_history' column
data['smoking_history'] = data['smoking_history'].map(smoking_history_mapping)

# Get Data type and count
data.info()

#Exploratory Data Analysis
#Get statistical report
print(data.describe(include ='all'))

# Plotting distributions of features
plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution')

plt.subplot(3, 3, 2)
plt.hist(data['bmi'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(True)


plt.subplot(3, 3, 3)
sns.histplot(data['HbA1c_level'], kde=True)
plt.title('HbA1c Level Distribution')

plt.subplot(3, 3, 4)
sns.histplot(data['blood_glucose_level'], kde=True)
plt.title('Blood Glucose Level Distribution')

plt.subplot(3, 3, 5)
# Count the occurrences of each gender
gender_counts = data['gender'].value_counts()
# Create a bar plot with different colors
plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'orange', 'green'], edgecolor='k', alpha=0.7)

plt.xticks([0, 1, 2], ['Female', 'Male', 'Other'])
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Histogram of Gender Distribution')
plt.grid(True)


plt.subplot(3, 3, 6)
bars = plt.bar(['No Hypertension', 'Hypertension'], data['hypertension'].value_counts(), color=['blue', 'orange'], edgecolor='black')
plt.title('Distribution of Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Frequency')
plt.grid(True)
plt.title('Hypertension Distribution')

plt.subplot(3, 3, 7)
bars = plt.bar(['No Heart Disease', 'Heart Disease'], data['heart_disease'].value_counts(), color=['green', 'red'], edgecolor='black')
plt.title('Distribution of Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.grid(True)

smoking_mapping = {0: 'never', 1: 'No Info', 2: 'current', 3: 'former', 4: 'not current', 5: 'ever'}
data['smoking_history'] = data['smoking_history'].map(smoking_mapping)

# Create a histogram for the 'smoking_history' column after mapping
plt.subplot(3, 3, 8)
bars = plt.bar(data['smoking_history'].value_counts().index, data['smoking_history'].value_counts(), color=['blue', 'orange', 'green', 'red', 'purple', 'brown'], edgecolor='black')
plt.title('Distribution of Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(True)


plt.subplot(3, 3, 9)
bars = plt.bar(['No Diabetes', 'Diabetes'], data['diabetes'].value_counts(), color=['blue', 'red'], edgecolor='black')
plt.title('Distribution of Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Frequency')
plt.grid(True)

# Age vs Diabetes
sns.boxplot(x='diabetes', y='age', data=data, ax=axs[0])
axs[0].set_title('Age vs Diabetes')
axs[0].set_xlabel('Diabetes')
axs[0].set_ylabel('Age')

# BMI vs Diabetes
sns.boxplot(x='diabetes', y='bmi', data=data, ax=axs[1])
axs[1].set_title('BMI vs Diabetes')
axs[1].set_xlabel('Diabetes')
axs[1].set_ylabel('BMI')

# Smoking History vs Diabetes
sns.countplot(x='diabetes', hue='smoking_history', data=data, ax=axs[2])
axs[2].set_title('Smoking History vs Diabetes')
axs[2].set_xlabel('Diabetes')
axs[2].set_ylabel('Count')
axs[2].legend(title='Smoking History', loc='upper right')

plt.tight_layout()
plt.show()

# Plot comparative charts for age and gender against the presence and absence of diabetes side by side

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Age vs Diabetes
sns.boxplot(x='diabetes', y='age', data=data, ax=axs[0])
axs[0].set_title('Age vs Diabetes')
axs[0].set_xlabel('Diabetes 0: No, 1: Yes')
axs[0].set_ylabel('Age')

# Gender vs Diabetes
sns.countplot(x='diabetes', hue='gender', data=data, ax=axs[1])
axs[1].set_title('Gender vs Diabetes')
axs[1].set_xlabel('Diabetes 0: No, 1: Yes')
axs[1].set_ylabel('Count')
axs[1].legend(title='Gender', loc='upper right')

plt.tight_layout()
plt.show()
# Plot comparative charts for BMI, smoking history, and hypertension against the presence and absence of diabetes

fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# BMI vs Diabetes
sns.boxplot(x='diabetes', y='bmi', data=data, ax=axs[0])
axs[0].set_title('BMI vs Diabetes')
axs[0].set_xlabel('Diabetes (0: No, 1: Yes)')
axs[0].set_ylabel('BMI')

# Smoking History vs Diabetes
sns.countplot(x='diabetes', hue='smoking_history', data=data, ax=axs[1])
axs[1].set_title('Smoking History vs Diabetes')
axs[1].set_xlabel('Diabetes (0: No, 1: Yes)')
axs[1].set_ylabel('Count')
axs[1].legend(title='Smoking History', loc='upper right')

# Hypertension vs Diabetes
sns.countplot(x='diabetes', hue='hypertension', data=data, ax=axs[2])
axs[2].set_title('Hypertension vs Diabetes')
axs[2].set_xlabel('Diabetes (0: No, 1: Yes)')
axs[2].set_ylabel('Count')
axs[2].legend(title='Hypertension', loc='upper right')

plt.tight_layout()
plt.show()

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of All Parameters')
plt.show()


#Machine Learning Models


# Random Forest

# Separate features and target variable
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Check if data is loaded correctly
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = pd.DataFrame(X_test, columns=feature_names)
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#Hyperparameter tuning of the Random Forest Classifier
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Save the model to a file
joblib.dump(rf_model, 'random_forest_model.pkl')


# Logistic Regression Model

# Split the data into features and target variable
X = data.drop(columns='diabetes')
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_rep)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve and calculate AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Logistic Regression model with L2 regularization


# Preprocess the data
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Define categorical and numerical columns
categorical_cols = ['gender', 'smoking_history']
numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model with L2 regularization
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l2', solver='liblinear'))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)


# Print accuracy and classification report
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_rep)

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Diabetes', 'Diabetes'], rotation=45)
plt.yticks(tick_marks, ['No Diabetes', 'Diabetes'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations
thresh = conf_matrix.max() / 2
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, conf_matrix[i, j],
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# Get feature importance
coefficients = model.named_steps['classifier'].coef_[0]
feature_names = numerical_cols + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols))
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})

# Sort by absolute value of importance
feature_importance['Abs_Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)

# Show the feature importance
feature_importance.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(12, 6))
plt.title('Feature Importance')
plt.ylabel('Coefficient')
plt.tight_layout()
plt.show()

# Support Vector Machine

# Split the dataset into features and target variable
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Support Vector Machine model
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)
y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy, precision, recall, f1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print accuracy, precision, recall, f1 score
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Print classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC/AUC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Optimized SVM

# Load the dataset
file_path = '/Users/user/Downloads/encoded_diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Split the data into features and target variable
X = data.drop(columns='diabetes')
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standard scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'log_reg__C': [0.1, 1, 10, 100],
    'log_reg__penalty': ['l2'],
    'log_reg__solver': ['lbfgs'],
    'log_reg__max_iter': [100, 200, 500]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions with the optimized model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate accuracy, precision, recall, and F1 score
optimized_accuracy = accuracy_score(y_test, y_pred)
optimized_precision = precision_score(y_test, y_pred)
optimized_recall = recall_score(y_test, y_pred)
optimized_f1 = f1_score(y_test, y_pred)
optimized_classification_rep = classification_report(y_test, y_pred)

# Print the metrics
print("Optimized Accuracy:", optimized_accuracy)
print("Optimized Precision:", optimized_precision)
print("Optimized Recall:", optimized_recall)
print("Optimized F1 Score:", optimized_f1)
print("\nOptimized Classification Report:\n", optimized_classification_rep)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Confusion Matrix')
plt.show()

# Plot ROC curve and calculate AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Optimized Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


#Neural Network
# Load the dataset
file_path = '/Users/user/Downloads/encoded_diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Prepare the data
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the feedforward neural network model
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=300, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Classification Report:\n{classification_rep}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Diabetes', 'Diabetes'], rotation=45)
plt.yticks(tick_marks, ['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Adding values to the confusion matrix plot
thresh = conf_matrix.max() / 2
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#Optimized Neural Network
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 300, 400]
}

# Initialize the model
model = MLPClassifier(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the best model
y_pred_prob_optimized = best_model.predict_proba(X_test)[:, 1]
y_pred_optimized = best_model.predict(X_test)

# Calculate metrics for the optimized model
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)
f1_optimized = f1_score(y_test, y_pred_optimized)
classification_rep_optimized = classification_report(y_test, y_pred_optimized)

# Print the best parameters and the metrics
print("Best Parameters:", best_params)
print(f"Optimized Accuracy: {accuracy_optimized}")
print(f"Optimized Precision: {precision_optimized}")
print(f"Optimized Recall: {recall_optimized}")
print(f"Optimized F1 Score: {f1_optimized}")
print(f"Optimized Classification Report:\n{classification_rep_optimized}")

# Plot confusion matrix for the optimized model
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix_optimized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Optimized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Diabetes', 'Diabetes'], rotation=45)
plt.yticks(tick_marks, ['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Adding values to the confusion matrix plot
thresh = conf_matrix_optimized.max() / 2
for i, j in itertools.product(range(conf_matrix_optimized.shape[0]), range(conf_matrix_optimized.shape[1])):
    plt.text(j, i, format(conf_matrix_optimized[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_optimized[i, j] > thresh else "black")
plt.show()

# Plot ROC curve for the optimized model
fpr_optimized, tpr_optimized, thresholds_optimized = roc_curve(y_test, y_pred_prob_optimized)
roc_auc_optimized = roc_auc_score(y_test, y_pred_prob_optimized)
plt.figure(figsize=(8, 6))
plt.plot(fpr_optimized, tpr_optimized, color='darkorange', lw=2, label=f'Optimized ROC curve (area = {roc_auc_optimized:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Optimized Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


#Web application built with Flask


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request
    features = np.array(data['features']).reshape(1, -1)  # Convert to numpy array and reshape for prediction
    prediction = model.predict(features)[0]  # Make the prediction
    result = "You are likely to have diabetes" if prediction == 1 else "You are not likely to have diabetes"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)


#HTML Web Code


<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: url('/static/background.jpg') no-repeat center center fixed;
            background-size: cover;
            color: #fff;
            text-align: center;
            padding: 50px;
        }
        h1 {
            font-size: 3em;
            margin-bottom: 20px;
        }
        form {
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
        }
        label {
            display: block;
            margin: 10px 0 5px;
        }
        input {
            width: 80%;
            padding: 10px;
            margin: 5px 0 20px;
            border: none;
            border-radius: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            font-size: 1.5em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Diabetes Prediction</h1>
    <form id="predictionForm">
        <label for="gender">gender:</label>
        <input type="number" id="gender" name="features" required><br><br>
        <label for="age">age:</label>
        <input type="number" id="age" name="features" required><br><br>
        <label for="hypertension">hypertension:</label>
        <input type="number" id="hypertension" name="features" required><br><br>
        <label for="heart_disease">heart_disease:</label>
        <input type="number" id="heart_disease" name="features" required><br><br>
        <label for="smoking_history">smoking_history:</label>
        <input type="number" id="smoking_history" name="features" required><br><br>
        <label for="bmi">bmi:</label>
        <input type="number" step="0.1" id="bmi" name="features" required><br><br>
        <label for="HbA1c_level">HbA1c_level:</label>
        <input type="number" step="0.001" id="HbA1c_level" name="features" required><br><br>
        <label for="blood_glucose_level">blood_glucose_level:</label>
        <input type="number" id="blood_glucose_level" name="features" required><br><br>
        <button type="submit">Predict</button>
    </form>
 <h2 id="result"></h2>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(event) {
            event.preventDefault();
            var formData = new FormData(this);
            var features = [];
            formData.forEach(function(value, key) {
                features.push(parseFloat(value));
            });

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: features })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = data.prediction;
            })
            .catch(error => console.error('Error:', error));
        });
    </script>
</body>
</html>
