#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
#%%

# This script demonstrates how to build a predictive model for waiting period compliance

def build_compliance_prediction_model(data_path):
    """
    Build a machine learning model to predict non-compliance with the 30-day waiting period

    Parameters:
    data_path (str): Path to the CSV file containing the synthetic data

    Returns:
    model (Pipeline): Trained sklearn pipeline
    X_test (DataFrame): Test features
    y_test (Series): Test target values
    """
    # Load data
    df = pd.read_csv(data_path)

    # Convert date strings to datetime objects and extract features
    df['DateOfRecipientSignature'] = pd.to_datetime(df['DateOfRecipientSignature'])
    df['Month'] = df['DateOfRecipientSignature'].dt.month
    df['Year'] = df['DateOfRecipientSignature'].dt.year
    df['DayOfWeek'] = df['DateOfRecipientSignature'].dt.dayofweek

    # Define features and target
    X = df[['SpecifyTypeOfOperation', 'RecipientSignature', 'RaceAndEthnicDesignation',
            'InterpreterSignature', 'Month', 'Year', 'DayOfWeek']]
    y = df['WaitingPeriodFlag']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessing pipeline
    categorical_features = ['SpecifyTypeOfOperation', 'RaceAndEthnicDesignation']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Model Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Compliant', 'Non-Compliant'],
                yticklabels=['Compliant', 'Non-Compliant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    # Calculate feature importance
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    forest = model.named_steps['classifier']

    # Get feature importances using permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Create a DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df[:15])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    print("\nTop 5 most important features:")
    print(importance_df.head(5))

    return model, X_test, y_test

#%%
# This function would be used with the actual generated data
"""
Example usage:
model, X_test, y_test = build_compliance_prediction_model('tenncare_sterilization_consent_synthetic_data.csv')

# Make predictions on new data
new_data = pd.DataFrame({
    'SpecifyTypeOfOperation': ['Bilateral Tubal Ligation'],
    'RecipientSignature': [1],
    'RaceAndEthnicDesignation': ['Hispanic'],
    'InterpreterSignature': [1],
    'Month': [6],
    'Year': [2023],
    'DayOfWeek': [2]
})

prediction = model.predict_proba(new_data)[:, 1]
print(f"Probability of non-compliance: {prediction[0]:.2%}")
"""

print("""
To build the prediction model:
1. Run the data generation script first to create the synthetic dataset
2. Run this script to train and evaluate the model
3. Use the model to predict which cases might be at risk of non-compliance

This model can help healthcare providers identify factors associated with non-compliance
and potentially implement targeted interventions to improve compliance rates.
""")
#%%
df = pd.read_csv('/Users/sayam_palrecha/my_project/Hackathon/train_data.csv')
# Convert date strings to datetime objects and extract features
df['DateOfRecipientSignature'] = pd.to_datetime(df['DateOfRecipientSignature'])
df['Month'] = df['DateOfRecipientSignature'].dt.month
df['Year'] = df['DateOfRecipientSignature'].dt.year
df['DayOfWeek'] = df['DateOfRecipientSignature'].dt.dayofweek

# Define features and target
X = df[['SpecifyTypeOfOperation', 'RecipientSignature', 'RaceAndEthnicDesignation',
        'InterpreterSignature', 'Month', 'Year', 'DayOfWeek']]
y = df['WaitingPeriodFlag']
#%%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
categorical_features = ['SpecifyTypeOfOperation', 'RaceAndEthnicDesignation']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
#%%
# Evaluate the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
#%%
print("Model Performance:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
#%%
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Compliant', 'Non-Compliant'],
            yticklabels=['Compliant', 'Non-Compliant'])

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
# plt.savefig('confusion_matrix.png')
#%%
# Calculate feature importance
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
forest = model.named_steps['classifier']
# Get feature importances using permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
# Create a DataFrame with feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)
#%%
# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df[:15])
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nTop 5 most important features:")
print(importance_df.head(5))
#%%
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to 'model.pkl'")