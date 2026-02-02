import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Explore missing values (consider imputation instead of dropping)
print("Missing values summary:")
print(df.isnull().sum())

# Handle missing values (e.g., imputation) - TBD

# Feature scaling
scaler = StandardScaler()
features = df.drop('DEATH_EVENT', axis=1)  # Target variable

# Identify key numeric features
scaled_features = scaler.fit_transform(features)

# PCA with 95% variance explained
pca = PCA(n_components=0.95)
pca_df = pca.fit_transform(scaled_features)

# Explained variance ratio for all components
explained_variance = pca.explained_variance_ratio_

# Top 5 components and their explained variance
top_5_components = explained_variance[:5].cumsum()

x_positions = [1, 2, 3, 4, 5]

# CDF plot for top 5 components' explained variance
plt.plot(range(1, 6), top_5_components, label='Top 5 components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Cumulative Explained Variance by Top 5 components')
plt.xticks(x_positions)
plt.legend()
plt.grid(True)
plt.show()

# Train-Test Split for classification
X_train, X_test, y_train, y_test = train_test_split(pca_df, df['DEATH_EVENT'], test_size=0.2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on testing set
y_pred = model.predict(X_test)

# Evaluate model performance (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Format accuracy to 4 decimal places

# Print a sample of actual vs. predicted values (you can adjust the number of samples)
num_samples = 10
print("\nActual vs. Predicted (Sample):")
for i in range(num_samples):
    print(f"Sample {i+1}: Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

# Comparison of y_pred and y_actual:
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Loop through top 5 components
# Top features contributing to top components (approximate)
print("\nTop Features Contributing to Top Principal Components (Approximate):")
component_names = [f"Component {i+1}" for i in range(pca.n_components_)]  # Create component names

# Loop through top 5 components
for i in range(5):
    # Get absolute loadings for the i-th component
    abs_loadings = abs(pca.components_[i])
    # Normalize loadings to percentages
    percentage_loadings = (abs_loadings / abs_loadings.sum()) * 100
    # Sort features by contribution (descending order)
    sorted_features = sorted(zip(features.columns, percentage_loadings), key=lambda x: x[1], reverse=True)

    # Print component name
    print(f"\nComponent {i+1}:")

    # Print top 3 features and contributions
    for j in range(3):
        if j >= len(sorted_features):
            break  # Handle cases with less than 3 features
        feature_name, contribution = sorted_features[j]
        print(f"\t{feature_name}: {contribution:.2f}%")