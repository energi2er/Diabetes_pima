import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
diabetes_data = pd.read_csv('/home/farid/Diabetes_pima/diabetes.csv')

# Step 1: Split your data
X = diabetes_data.drop('Outcome', axis=1)  # Features
y = diabetes_data['Outcome']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Perform feature engineering only on train data
# Define selected features
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Generate combinations of features
feature_combinations = list(itertools.combinations(selected_features, 2))

# Iterate over feature combinations
for combination in feature_combinations:
    # Generate a new feature name
    new_feature_name = '_'.join(combination)
    
    # Calculate the new feature values
    new_feature_values = X_train[list(combination)].sum(axis=1)
    
    # Add the new feature to the DataFrame
    X_train[new_feature_name] = new_feature_values

# Step 3: Visualize distribution of features (using only train data)
#sns.pairplot(X_train, hue=y_train.astype(str))
#plt.show()


# Step 4: Explore correlations (using only train data)
correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Step 5: Continue with model training and evaluation
# Step 6: Train your model
# For example, let's train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter value if needed
logistic_model.fit(X_train, y_train)

# Step 7: Evaluate your model
# Predict on test data
y_pred = logistic_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
