
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=============================================")
print("TELECOM CUSTOMER CHURN ANALYSIS")
print("=============================================")

print("\n1. DATA LOADING AND EXPLORATION")
print("---------------------------------------------")

# Load the data
df = pd.read_csv('telco_churn_data.csv')

# Basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Data types and missing values
print("\nData types and missing values:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

missing_values = df.isnull().sum()
print("\nMissing values by column:")
print(missing_values[missing_values > 0])

print("\n2. DATA PREPROCESSING")
print("---------------------------------------------")

# For numeric columns i used median imputation
# For categorical columns i used mode imputation
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

if 'Customer ID' in numeric_features:
    numeric_features.remove('Customer ID')
if 'Customer ID' in categorical_features:
    categorical_features.remove('Customer ID')
if 'Churn Value' in numeric_features:
    numeric_features.remove('Churn Value')

if 'Churn Value' in df.columns:
    df['Churn Value'] = df['Churn Value'].astype(int)

print("\nTarget variable distribution:")
print(df['Churn Value'].value_counts())
print(f"Churn Rate: {df['Churn Value'].mean():.2%}")

X = df.drop(['Customer ID', 'Churn Value', 'Churn Category', 'Churn Reason'], axis=1, errors='ignore')
y = df['Churn Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

numeric_features = [col for col in numeric_features if col in X.columns]
categorical_features = [col for col in categorical_features if col in X.columns]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\n3. EXPLORATORY DATA ANALYSIS")
print("---------------------------------------------")

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Contract', hue='Churn Value', data=df)
plt.title('Churn by Contract Type')

plt.subplot(2, 2, 2)
sns.countplot(x='Internet Type', hue='Churn Value', data=df)
plt.title('Churn by Internet Type')

plt.subplot(2, 2, 3)
sns.countplot(x='Payment Method', hue='Churn Value', data=df)
plt.title('Churn by Payment Method')

plt.subplot(2, 2, 4)
sns.countplot(x='Paperless Billing', hue='Churn Value', data=df)
plt.title('Churn by Paperless Billing')

plt.tight_layout()
plt.savefig('churn_by_categories.png')

# Plot numeric features distribution by churn
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.boxplot(x='Churn Value', y='Monthly Charge', data=df)
plt.title('Monthly Charge by Churn')

plt.subplot(2, 2, 2)
sns.boxplot(x='Churn Value', y='Tenure in Months', data=df)
plt.title('Tenure by Churn')

plt.subplot(2, 2, 3)
sns.boxplot(x='Churn Value', y='Total Customer Svc Requests', data=df)
plt.title('Customer Service Requests by Churn')

plt.subplot(2, 2, 4)
sns.boxplot(x='Churn Value', y='Age', data=df)
plt.title('Age by Churn')

plt.tight_layout()
plt.savefig('churn_by_numerics.png')

# Feature correlation analysis
plt.figure(figsize=(16, 12))
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.savefig('correlation_matrix.png')

print("\n4. MODEL BUILDING AND EVALUATION")
print("---------------------------------------------")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    if hasattr(pipeline.named_steps['model'], 'coef_') or hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        get_feature_importance(pipeline, model_name, X_train)
    
    return pipeline, accuracy, roc_auc, fpr, tpr

def get_feature_importance(pipeline, model_name, X):
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # Get feature names
    feature_names = []
    for transformer_name, transformer, column_names in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            # For newer scikit-learn versions
            try:
                feature_names.extend(
                    [f"{name}" for name in transformer.get_feature_names_out(column_names)]
                )
            except:
                pass
    
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return
    
    if len(importances) == len(feature_names):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_feature_importance.png')
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    pipeline, accuracy, roc_auc, fpr, tpr = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')

print("\n5. MODEL IMPROVEMENT")
print("---------------------------------------------")

best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
print(f"Best model based on ROC AUC: {best_model_name}")

param_grids = {
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear']
    },
    'Decision Tree': {
        'model__max_depth': [3, 5, 7, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    },
    'KNN': {
        'model__n_neighbors': [3, 5, 7, 9, 11],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    }
}

best_model = models[best_model_name]
best_param_grid = param_grids[best_model_name]

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

grid_search = GridSearchCV(
    pipeline,
    param_grid=best_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the best model
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nImproved {best_model_name} Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

print("\n6. BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("---------------------------------------------")

print("\nTop 5 Churn Reasons:")
churn_reasons = df[df['Churn Value'] == 1]['Churn Reason'].value_counts().head(5)
print(churn_reasons)

print("\nChurn by Category:")
churn_categories = df[df['Churn Value'] == 1]['Churn Category'].value_counts()
print(churn_categories)

contract_churn = df.groupby('Contract')['Churn Value'].mean().sort_values(ascending=False)
print("\nChurn Rate by Contract Type:")
print(contract_churn)

df['Tenure Group'] = pd.cut(
    df['Tenure in Months'],
    bins=[0, 12, 24, 36, 48, 60, float('inf')],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+']
)
tenure_churn = df.groupby('Tenure Group')['Churn Value'].mean().sort_values(ascending=False)
print("\nChurn Rate by Tenure Group:")
print(tenure_churn)

# Plot churn rate by tenure group
plt.figure(figsize=(10, 6))
sns.barplot(x=tenure_churn.index, y=tenure_churn.values)
plt.title('Churn Rate by Tenure Group')
plt.ylabel('Churn Rate')
plt.savefig('churn_by_tenure.png')

# Plot churn reasons distribution
plt.figure(figsize=(12, 8))
sns.countplot(y='Churn Reason', data=df[df['Churn Value'] == 1], order=df[df['Churn Value'] == 1]['Churn Reason'].value_counts().index[:10])
plt.title('Top 10 Churn Reasons')
plt.tight_layout()
plt.savefig('top_churn_reasons.png')

# Plot churn rate by monthly charge groups
df['Monthly Charge Group'] = pd.cut(
    df['Monthly Charge'],
    bins=[0, 50, 75, 100, 125, float('inf')],
    labels=['$0-$50', '$50-$75', '$75-$100', '$100-$125', '$125+']
)
charge_churn = df.groupby('Monthly Charge Group')['Churn Value'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=charge_churn.index, y=charge_churn.values)
plt.title('Churn Rate by Monthly Charge Group')
plt.ylabel('Churn Rate')
plt.savefig('churn_by_charge.png')

print("\n=============================================")
print("ANALYSIS COMPLETE - Check generated visualizations")
print("=============================================") 