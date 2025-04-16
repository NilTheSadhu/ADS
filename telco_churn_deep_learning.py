import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=============================================")
print("TELECOM CUSTOMER CHURN ANALYSIS WITH DEEP LEARNING")
print("=============================================")

print("\n1. DATA LOADING AND PREPROCESSING")
print("---------------------------------------------")

# Load the data
df = pd.read_csv('telco_churn_data.csv')

# Basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

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
    # This is a simplified example
    df['County'] = df['Zip Code'].astype(str).str[:2]
    print("\nEngineered new feature: County from ZIP codes")
    print(f"Reduced cardinality from {df['Zip Code'].nunique()} to {df['County'].nunique()}")

if 'Churn Value' in df.columns:
    df['Churn Value'] = df['Churn Value'].astype(int)

print("\nTarget variable distribution:")
print(df['Churn Value'].value_counts())
print(f"Churn Rate: {df['Churn Value'].mean():.2%}")

# Define feature types
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

if 'Customer ID' in numeric_features:
    numeric_features.remove('Customer ID')
if 'Customer ID' in categorical_features:
    categorical_features.remove('Customer ID')
if 'Churn Value' in numeric_features:
    numeric_features.remove('Churn Value')

# Prepare data for modeling
X = df.drop(['Churn Value'], axis=1, errors='ignore')
y = df['Churn Value']

# Update feature lists after preprocessing
numeric_features = [col for col in numeric_features if col in X.columns]
categorical_features = [col for col in categorical_features if col in X.columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
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

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)
X_test_tensor = torch.FloatTensor(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

print(f"\nProcessed feature dimensions: {X_train_tensor.shape[1]}")

# Create dataset class
class TelcoChurnDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create data loaders
batch_size = 64
train_dataset = TelcoChurnDataset(X_train_tensor, y_train_tensor)
test_dataset = TelcoChurnDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n2. DEEP LEARNING MODELS")
print("---------------------------------------------")

# 1. Basic Neural Network (Multilayer Perceptron)
class BasicNN(nn.Module):
    def __init__(self, input_dim):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# 2. Deep Neural Network
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x

# 3. Residual Neural Network
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim):
        super(ResNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, 128)
        self.bn_in = nn.BatchNorm1d(128)
        self.residual_block1 = ResidualBlock(128)
        self.residual_block2 = ResidualBlock(128)
        self.fc_out = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = torch.sigmoid(self.fc_out(x))
        return x

# 4. Self-Attention based model
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        query = self.query(x)  # (batch_size, 1, input_dim)
        key = self.key(x).transpose(1, 2)  # (batch_size, input_dim, 1)
        energy = torch.bmm(query, key)  # (batch_size, 1, 1)
        attention = F.softmax(energy, dim=2)  # (batch_size, 1, 1)
        
        value = self.value(x)  # (batch_size, 1, input_dim)
        out = torch.bmm(attention, value)  # (batch_size, 1, input_dim)
        out = out.squeeze(1)  # (batch_size, input_dim)
        
        return self.gamma * out + x.squeeze(1)

class AttentionNN(nn.Module):
    def __init__(self, input_dim):
        super(AttentionNN, self).__init__()
        self.attention = SelfAttention(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    model = model.to(device)
    best_val_auc = 0.0
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            y_true.extend(targets.cpu().detach().numpy())
            y_pred.extend(outputs.cpu().detach().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Calculate training AUC
        train_auc = roc_auc_score(y_true, y_pred)
        train_aucs.append(train_auc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation AUC
        val_auc = roc_auc_score(y_true, y_pred)
        val_aucs.append(val_auc)
        
        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        else:
            no_improve_epochs += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Check early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label='Train AUC')
    plt.plot(val_aucs, label='Validation AUC')
    plt.title('AUC Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model.__class__.__name__}_training_history.png")
    
    # Load the best model
    model.load_state_dict(torch.load(f"{model.__class__.__name__}_best.pth"))
    return model, best_val_auc

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            
            predicted = (outputs > 0.5).float()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model.__class__.__name__}')
    plt.legend(loc="lower right")
    plt.savefig(f"{model.__class__.__name__}_roc_curve.png")
    
    return accuracy, roc_auc, report, conf_matrix, fpr, tpr

# Train and evaluate models
input_dim = X_train_tensor.shape[1]
models = {
    "Basic Neural Network": BasicNN(input_dim),
    "Deep Neural Network": DeepNN(input_dim),
    "Residual Neural Network": ResNet(input_dim),
    "Attention Neural Network": AttentionNN(input_dim)
}

results = {}
criterion = nn.BCELoss()
learning_rate = 0.001
weight_decay = 1e-5

# Initialize plot for ROC curves
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', lw=2)

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, best_val_auc = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, patience=10)
    
    print(f"\nEvaluating {name}...")
    accuracy, roc_auc, report, conf_matrix, fpr, tpr = evaluate_model(model, test_loader)
    
    training_time = time.time() - start_time
    
    results[name] = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "training_time": training_time,
        "fpr": fpr,
        "tpr": tpr
    }
    
    print(f"\nResults for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Add to ROC plot
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

# Finalize ROC plot
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Different Deep Learning Models', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('deep_learning_roc_curves.png', dpi=300, bbox_inches='tight')

# Compare model performance
print("\n3. MODEL COMPARISON")
print("---------------------------------------------")

# Create a comparison table
model_names = list(results.keys())
accuracies = [results[model]["accuracy"] for model in model_names]
roc_aucs = [results[model]["roc_auc"] for model in model_names]
training_times = [results[model]["training_time"] for model in model_names]

comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'ROC AUC': roc_aucs,
    'Training Time (s)': training_times
})

print("\nModel Comparison:")
print(comparison_df.sort_values('ROC AUC', ascending=False))

# Plot comparison
plt.figure(figsize=(12, 6))

# Accuracy and AUC comparison
plt.subplot(1, 2, 1)
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, accuracies, bar_width, label='Accuracy')
plt.bar(index + bar_width, roc_aucs, bar_width, label='ROC AUC')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width/2, [name.split()[0] for name in model_names], rotation=45)
plt.legend()

# Training time comparison
plt.subplot(1, 2, 2)
plt.bar(index, training_times)
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.xticks(index, [name.split()[0] for name in model_names], rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png')

print("\n4. FEATURE IMPORTANCE ANALYSIS")
print("---------------------------------------------")

# Since direct feature importance is not available in neural networks,
# we'll use a simple permutation importance approach

def permutation_importance(model, X, y, n_repeats=10):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    # Get baseline score
    with torch.no_grad():
        baseline_preds = model(X_tensor).squeeze().cpu().numpy()
    baseline_score = roc_auc_score(y, baseline_preds)
    
    # Get feature names (columns after preprocessing)
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Calculate importance for each feature
    importances = []
    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            # Create shuffled data
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Score with shuffled feature
            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
            with torch.no_grad():
                preds = model(X_permuted_tensor).squeeze().cpu().numpy()
            score = roc_auc_score(y, preds)
            
            # Calculate importance
            scores.append(baseline_score - score)
        
        # Average importance over repeats
        importances.append(np.mean(scores))
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    return importance_df.sort_values('Importance', ascending=False)

# Get the best model based on ROC AUC
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = models[best_model_name]

# Calculate feature importance
X_test_numpy = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed
importance_df = permutation_importance(best_model, X_test_numpy, y_test.values)

print(f"\nFeature Importance for {best_model_name}:")
print(importance_df.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title(f'Top 15 Feature Importances - {best_model_name}')
plt.tight_layout()
plt.savefig('deep_learning_feature_importance.png')

print("\n=============================================")
print("DEEP LEARNING ANALYSIS COMPLETE")
print("=============================================") 