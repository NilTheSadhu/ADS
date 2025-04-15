import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
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

# Check for duplicate records
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate records.")

# Calculate cardinality of each feature
cardinality = {col: df[col].nunique() for col in df.columns}
print("\nCardinality of each feature:")
for col, count in cardinality.items():
    print(f"{col}: {count} unique values")

# Identify high-cardinality features (>10% of dataset size)
high_cardinality_features = [col for col in cardinality if cardinality[col] > 0.1 * len(df) and col != 'Churn Value']
print("\nHigh cardinality features to remove:")
print(high_cardinality_features)

# Remove high-cardinality features like City and Customer ID
df = df.drop(columns=high_cardinality_features, errors='ignore')

# Check for features with excessive missing values (>30%)
missing_pct = df.isnull().mean() * 100
excessive_missing = missing_pct[missing_pct > 30].index.tolist()
print("\nFeatures with excessive missing values (>30%):")
print(excessive_missing)

# Drop features with excessive missing values that cannot be imputed
features_to_drop = ['Churn Category', 'Churn Reason', 'Customer Satisfaction', 'Offer']
df = df.drop(columns=features_to_drop, errors='ignore')
print(f"\nDropped features with excessive missing values: {features_to_drop}")

# Insert feature values where appropriate
# For example, fill missing Internet Type for customers not paying for Internet Service
if 'Internet Type' in df.columns and 'Internet Service' in df.columns:
    internet_service_mask = df['Internet Service'] == 'No'
    if df['Internet Type'].isnull().any():
        df.loc[internet_service_mask, 'Internet Type'] = 'No Internet Service'
        print("\nFilled missing Internet Type values for customers without Internet Service")

# Engineer new feature - County from ZIP codes
if 'Zip Code' in df.columns:
    # This is a simplified example - in reality you would need a mapping from ZIP codes to counties
    # Here we're just taking the first two digits of the ZIP code as a proxy for county
    df['County'] = df['Zip Code'].astype(str).str[:2]
    print("\nEngineered new feature: County from ZIP codes")
    print(f"Reduced cardinality from {df['Zip Code'].nunique()} to {df['County'].nunique()}")

# Check for outliers in numeric columns
print("\nChecking for outliers in numeric columns:")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Churn Value' in numeric_cols:
    numeric_cols.remove('Churn Value')
if 'Customer ID' in numeric_cols:
    numeric_cols.remove('Customer ID')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        outlier_pct = len(outliers) / len(df) * 100
        print(f"{col}: {len(outliers)} outliers ({outlier_pct:.2f}% of data)")

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

# Prepare data for modeling
X = df.drop(['Churn Value'], axis=1, errors='ignore')
y = df['Churn Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Update feature lists after preprocessing
numeric_features = [col for col in numeric_features if col in X.columns]
categorical_features = [col for col in categorical_features if col in X.columns]

# Changed from StandardScaler to MinMaxScaler as requested
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())  # Changed to MinMaxScaler
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

# Regenerate all the visualizations
# First, let's check distributions of features
print("\nChecking distributions of key features:")
for col in numeric_features[:5]:  # Show first 5 numeric features
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Churn Value', y=col, data=df)
    plt.title(f'{col} by Churn')
    
    plt.tight_layout()
    plt.savefig(f'distribution_{col}.png')
    plt.close()

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Contract', hue='Churn Value', data=df)
plt.title('Churn by Contract Type')

plt.subplot(2, 2, 2)
if 'Internet Type' in df.columns:
    sns.countplot(x='Internet Type', hue='Churn Value', data=df)
    plt.title('Churn by Internet Type')

plt.subplot(2, 2, 3)
if 'Payment Method' in df.columns:
    sns.countplot(x='Payment Method', hue='Churn Value', data=df)
    plt.title('Churn by Payment Method')

plt.subplot(2, 2, 4)
if 'Paperless Billing' in df.columns:
    sns.countplot(x='Paperless Billing', hue='Churn Value', data=df)
    plt.title('Churn by Paperless Billing')

plt.tight_layout()
plt.savefig('churn_by_categories.png')

# Plot numeric features distribution by churn
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
if 'Monthly Charge' in df.columns:
    sns.boxplot(x='Churn Value', y='Monthly Charge', data=df)
    plt.title('Monthly Charge by Churn')

plt.subplot(2, 2, 2)
if 'Tenure in Months' in df.columns:
    sns.boxplot(x='Churn Value', y='Tenure in Months', data=df)
    plt.title('Tenure by Churn')

plt.subplot(2, 2, 3)
if 'Total Customer Svc Requests' in df.columns:
    sns.boxplot(x='Churn Value', y='Total Customer Svc Requests', data=df)
    plt.title('Customer Service Requests by Churn')

plt.subplot(2, 2, 4)
if 'Age' in df.columns:
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

# If County feature was created, analyze churn by county
if 'County' in df.columns:
    # Get top 10 counties by frequency
    top_counties = df['County'].value_counts().head(10).index
    county_df = df[df['County'].isin(top_counties)]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='County', y='Churn Value', data=county_df, estimator=lambda x: np.mean(x) * 100)
    plt.title('Churn Rate by County (Top 10 Counties)')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('churn_by_county.png')

print("\n4. MODEL BUILDING AND EVALUATION")
print("---------------------------------------------")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )

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

    if hasattr(pipeline.named_steps["model"], "coef_") or hasattr(
        pipeline.named_steps["model"], "feature_importances_"
    ):
        get_feature_importance(pipeline, model_name, X_train)

    return pipeline, accuracy, roc_auc, fpr, tpr


def get_feature_importance(pipeline, model_name, X):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Get feature names
    feature_names = []
    for transformer_name, transformer, column_names in preprocessor.transformers_:
        if hasattr(transformer, "get_feature_names_out"):
            # For newer scikit-learn versions
            try:
                feature_names.extend(
                    [
                        f"{name}"
                        for name in transformer.get_feature_names_out(
                            column_names
                        )
                    ]
                )
            except:
                pass

    if hasattr(model, "coef_"):
        importances = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    if len(importances) == len(feature_names):
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )

        importance_df = importance_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="Importance", y="Feature", data=importance_df.head(15)
        )
        plt.title(f"Top 15 Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_feature_importance.png')

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))


results = {}
# Create a new figure with a white background and clear styling
plt.figure(figsize=(12, 8), facecolor='white')
plt.clf()  # Clear the current figure

# Set a clean style with minimal grid lines
plt.style.use('default')  # Reset to default style
plt.grid(alpha=0.3, linestyle='--')  # Subtle grid

for name, model in models.items():
    pipeline, accuracy, roc_auc, fpr, tpr = evaluate_model(
        name, model, X_train, X_test, y_train, y_test
    )
    results[name] = {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
    }

    # Plot each ROC curve with clear colors
    if name == "Decision Tree":
        color = 'blue'
    elif name == "SVM":
        color = 'red'
    elif name == "KNN":
        color = 'green'
    else:
        color = 'orange'
        
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

# Set clear axis limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Add labels and title
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for Different Models", fontsize=14)

# Add alternative axis labels as text
plt.text(-0.08, 0.5, 'Customer Satisfaction', fontsize=12, 
         rotation=90, ha='center', va='center')
plt.text(0.5, -0.08, 'Total Customer Svc Requests', fontsize=12, 
         ha='center', va='center')

# Add legend
plt.legend(loc="lower right", fontsize=10)

# Ensure tight layout and clean margins
plt.tight_layout()

# Save the figure with high resolution and transparency
plt.savefig("roc_curves.png", dpi=300, bbox_inches='tight', transparent=False)


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

# Since we've removed Churn Category and Churn Reason, we'll skip those analyses
# and focus on the features we have

if 'Contract' in df.columns:
    contract_churn = df.groupby('Contract')['Churn Value'].mean().sort_values(ascending=False)
    print("\nChurn Rate by Contract Type:")
    print(contract_churn)

if 'Tenure in Months' in df.columns:
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

# Since we removed Churn Reason, we'll skip the churn reasons plot
# Instead, let's create additional insightful plots

# Plot churn rate by internet type if available
if 'Internet Type' in df.columns:
    internet_churn = df.groupby('Internet Type')['Churn Value'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=internet_churn.index, y=internet_churn.values)
    plt.title('Churn Rate by Internet Type')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('churn_by_internet_type.png')

# Plot churn rate by payment method if available
if 'Payment Method' in df.columns:
    payment_churn = df.groupby('Payment Method')['Churn Value'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=payment_churn.index, y=payment_churn.values)
    plt.title('Churn Rate by Payment Method')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('churn_by_payment_method.png')

# Plot churn rate by monthly charge groups
if 'Monthly Charge' in df.columns:
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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Age distribution by churn
if 'Age' in df.columns:
    sns.histplot(data=df, x='Age', hue='Churn Value', kde=True, ax=axes[0])
    axes[0].set_title('Age Distribution by Churn')

# Gender churn rates
if 'Gender' in df.columns:
    gender_churn = df.groupby('Gender')['Churn Value'].mean().reset_index()
    sns.barplot(data=gender_churn, x='Gender', y='Churn Value', ax=axes[1])
    axes[1].set_title('Churn Rate by Gender')
    axes[1].set_ylabel('Churn Rate')

# Dependents impact
if 'Dependents' in df.columns:
    deps_churn = df.groupby('Dependents')['Churn Value'].mean().reset_index()
    sns.barplot(data=deps_churn, x='Dependents', y='Churn Value', ax=axes[2])
    axes[2].set_title('Churn Rate by Dependents')
    axes[2].set_ylabel('Churn Rate')

plt.tight_layout()
plt.savefig('demographic_churn_patterns.png')

# Generate visualization for the newly created County feature
if 'County' in df.columns and 'Tenure Bin' in df.columns:
    # Create a heatmap of churn rates by county and tenure
    county_tenure_pivot = df.groupby(['County', 'Tenure Bin'])['Churn Value'].mean().unstack()
    
    # Select top counties by number of customers
    top_counties = df['County'].value_counts().head(10).index
    county_tenure_pivot = county_tenure_pivot.loc[top_counties]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(county_tenure_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Churn Rate by County and Tenure')
    plt.tight_layout()
    plt.savefig('county_tenure_churn_heatmap.png')

# Code to generate data for tenure and contract analysis
if 'Tenure in Months' in df.columns and 'Contract' in df.columns:
    tenure_bins = range(0, 73, 12)
    tenure_labels = [f'{i}-{i+11}' for i in tenure_bins[:-1]] + ['72+']
    
    if 'Tenure Bin' not in df.columns:
        df['Tenure Bin'] = pd.cut(df['Tenure in Months'], 
                               bins=[*tenure_bins, float('inf')], 
                               labels=tenure_labels)
    
    tenure_contract_churn = df.groupby(['Tenure Bin', 'Contract'])['Churn Value'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tenure_contract_churn, x='Tenure Bin', y='Churn Value', hue='Contract')
    plt.title('Churn Rate by Tenure and Contract Type')
    plt.ylabel('Churn Rate')
    plt.savefig('tenure_contract_churn.png')