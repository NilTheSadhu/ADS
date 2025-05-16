import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, recall_score
import time
from uszipcode import SearchEngine
import optuna
import plotly
import kaleido
import os
from imblearn.over_sampling import SMOTE
import ast
#Note use: sqlalchemy-mate==1.4.28.4 uszipcode==1.0.1
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=============================================")
print("TELECOM CUSTOMER CHURN ANALYSIS WITH DEEP LEARNING")
print("=============================================")

print("DATA LOADING AND PREPROCESSING")
print("---------------------------------------------")

# Load the data
df = pd.read_csv('telco_churn_data.csv')

# Basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

#prepping directories for output:
os.makedirs('model_outputs', exist_ok=True)
os.makedirs('hpo_visualizations', exist_ok=True)
#Verifying presence of duplicates
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate records.")
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
#Based upon investigation we know that na in internet type corresponds to no for internet service
df['Internet Type'].fillna('Not Applicable', inplace=True)
cardinality = {col: df[col].nunique() for col in df.columns if col in categorical_features}
print("\nCardinality of each categorical feature:")
for col, count in cardinality.items():
    print(f"{col}: {count} unique values")
# Engineer new feature - County from ZIP codes
search = SearchEngine()
def get_county_by_zip(zip_code):
    result = search.by_zipcode(zip_code)
    return result.county if result else print("error")
df['County']=df['Zip Code'].apply(get_county_by_zip)
print(df['County'].nunique()) #58 unique counties - reasonable for 7000+ samples
print("\nEngineered new feature: County from ZIP codes")
print(f"Reduced cardinality from {df['Zip Code'].nunique()} to {df['County'].nunique()}")
df.drop(columns=["Zip Code","Longitude", "Latitude"],inplace=True)
print("Dropping columns: Zip Code, Longitude, Latitude")

# Convert target variable to binary
if 'Churn Value' in df.columns:
    df['Churn Value'] = df['Churn Value'].astype(int)

# Identify high-cardinality features (>10% of dataset size)
high_cardinality_features = [col for col in cardinality if cardinality[col] > 0.1 * len(df) and col != 'Churn Value']
print("\nHigh cardinality features to remove:")
print(high_cardinality_features)

# Remove high-cardinality features like City and Customer ID
print("Updated high cardinality features for removal: ",high_cardinality_features)
df = df.drop(columns=high_cardinality_features, errors='ignore')

# Check for features with excessive missing values (>30%)
missing_pct = df.isnull().mean() * 100
excessive_missing = missing_pct[missing_pct > 5].index.tolist()
print("\nFeatures with excessive missing values (>30%):")
print(excessive_missing)
df.drop(columns=excessive_missing, inplace=True)

try:
  df.drop(columns=['Churn Category','Churn Reason','Customer Satisfaction','Offer'],inplace=True)
except:
  print("No columns to drop - Already dropped")
print(f"\nDropped features with excessive missing values")



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


#Updated portion with optuna tuning and 3-way split
X_train_full, X_test_orig, y_train_full, y_test_orig = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y # Hold out 20% for final testing
)

# Split Full Training set into HPO-Training and HPO-Validation sets
X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full # 0.25 * 0.8 = 0.2 of total for HPO val
)

print(f"\nFull training set shape: {X_train_full.shape}")
print(f"HPO training set shape: {X_train_hpo.shape}")
print(f"HPO validation set shape: {X_val_hpo.shape}")
print(f"Test set shape: {X_test_orig.shape}")


# Define preprocessing steps (This is the active preprocessor for HPO and final models)
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()) # MinMaxScaler is generally good for NNs
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary')) # MODIFIED HERE
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough') # In case some features were missed, though ideally all are covered


# Define feature types explicitly after all engineering steps
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Explicitly ensure County is categorical
if 'County' not in categorical_features:
    print("Appended")
    categorical_features.append('County')

# Remove target and IDs explicitly if present
for col in ['Churn Value', 'Customer ID']:
    if col in categorical_features:
        categorical_features.remove(col)
    if col in numeric_features:
        numeric_features.remove(col)

# Final check
print("\nFinal categorical features:")
print(categorical_features)

print("\nFinal numeric features:")
print(numeric_features)

# Define preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough'
)

# Fit preprocessor on full training set
preprocessor.fit(X_train_full)

# Verify features after preprocessing
feature_names_processed = preprocessor.get_feature_names_out()
print("\nFeature names after preprocessing:")
print(feature_names_processed)
print(f"Total features after preprocessing: {len(feature_names_processed)}")

# Transform all datasets
X_train_hpo_processed = preprocessor.transform(X_train_hpo)
X_train_hpo_processed, y_train_hpo = SMOTE().fit_resample(X_train_hpo_processed, y_train_hpo)
X_val_hpo_processed = preprocessor.transform(X_val_hpo)
X_val_hpo_processed, y_val_hpo = SMOTE().fit_resample(X_val_hpo_processed, y_val_hpo)

X_train_full_processed = preprocessor.transform(X_train_full) # For final model training
X_test_processed = preprocessor.transform(X_test_orig)
X_train_full_processed, y_train_full = SMOTE().fit_resample(X_train_full_processed, y_train_full)

try:
    feature_names_processed = preprocessor.get_feature_names_out()
except Exception as e:
    print(f"Could not get feature names from preprocessor: {e}. Using generic names for importance.")
    feature_names_processed = [f"feature_{i}" for i in range(X_train_hpo_processed.shape[1])]


# Convert to PyTorch tensors
X_train_hpo_tensor = torch.FloatTensor(X_train_hpo_processed)
y_train_hpo_tensor = torch.FloatTensor(y_train_hpo.values)
X_val_hpo_tensor = torch.FloatTensor(X_val_hpo_processed)
y_val_hpo_tensor = torch.FloatTensor(y_val_hpo.values)

X_train_full_tensor = torch.FloatTensor(X_train_full_processed) # For final training
y_train_full_tensor = torch.FloatTensor(y_train_full.values)   # For final training

X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test_orig.values)


print(f"\nProcessed HPO training feature dimensions: {X_train_hpo_tensor.shape[1]}")
print(f"Processed HPO validation feature dimensions: {X_val_hpo_tensor.shape[1]}")
print(f"Processed Test feature dimensions: {X_test_tensor.shape[1]}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.90, gamma=1.5).to(device)
class TelcoChurnDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets.unsqueeze(1) # Ensure target is [batch_size, 1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

batch_size = 64

train_hpo_dataset = TelcoChurnDataset(X_train_hpo_tensor, y_train_hpo_tensor)
val_hpo_dataset = TelcoChurnDataset(X_val_hpo_tensor, y_val_hpo_tensor)
train_full_dataset = TelcoChurnDataset(X_train_full_tensor, y_train_full_tensor) # For final model
test_dataset = TelcoChurnDataset(X_test_tensor, y_test_tensor)

train_hpo_loader = DataLoader(train_hpo_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
val_hpo_loader = DataLoader(val_hpo_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
train_full_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True,drop_last=True) #for final model
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

print("\nDEEP LEARNING MODELS AND HYPERPARAMETER OPTIMIZATION")
print("---------------------------------------------")

# --- Model Definitions (Copied from your code, with minor potential adjustments for HPO) ---
class BasicNN(nn.Module):
    def __init__(self, input_dim, n_units_l1=128, n_units_l2=64, dropout_l1=0.3, dropout_l2=0.3):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_units_l1)
        self.bn1 = nn.BatchNorm1d(n_units_l1)
        self.dropout1 = nn.Dropout(dropout_l1)
        self.fc2 = nn.Linear(n_units_l1, n_units_l2)
        self.bn2 = nn.BatchNorm1d(n_units_l2)
        self.dropout2 = nn.Dropout(dropout_l2)
        self.fc3 = nn.Linear(n_units_l2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)  # Updated
        return x

class DeepNN(nn.Module):
    def __init__(self, input_dim, n_units_l1=256, n_units_l2=128, n_units_l3=64, n_units_l4=32, dropout_l1=0.4, dropout_l2=0.4, dropout_l3=0.3, dropout_l4=0.2):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_units_l1)
        self.bn1 = nn.BatchNorm1d(n_units_l1)
        self.dropout1 = nn.Dropout(dropout_l1)
        self.fc2 = nn.Linear(n_units_l1, n_units_l2)
        self.bn2 = nn.BatchNorm1d(n_units_l2)
        self.dropout2 = nn.Dropout(dropout_l2)
        self.fc3 = nn.Linear(n_units_l2, n_units_l3)
        self.bn3 = nn.BatchNorm1d(n_units_l3)
        self.dropout3 = nn.Dropout(dropout_l3)
        self.fc4 = nn.Linear(n_units_l3, n_units_l4)
        self.bn4 = nn.BatchNorm1d(n_units_l4)
        self.dropout4 = nn.Dropout(dropout_l4)
        self.fc5 = nn.Linear(n_units_l4, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)  # Updated
        return x
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_res=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_res)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out
class ResNet(nn.Module):
    def __init__(self, input_dim, initial_dim=128, res_block_dim=128, dropout_res=0.3, num_res_blocks=2):
        super(ResNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, initial_dim)
        self.bn_in = nn.BatchNorm1d(initial_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(initial_dim, dropout_res) for _ in range(num_res_blocks)])
        self.fc_out = nn.Linear(initial_dim, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.fc_out(x)  # Updated
        return x
class SelfAttention(nn.Module):
    def __init__(self, embed_size): # embed_size is the input feature dimension
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scalar

    def forward(self, x):
        x_unsqueeze = x.unsqueeze(1) # (batch_size, 1, embed_size)

        queries = self.query(x_unsqueeze) # (batch_size, 1, embed_size)
        keys = self.key(x_unsqueeze)    # (batch_size, 1, embed_size)
        values = self.value(x_unsqueeze) # (batch_size, 1, embed_size)

        # Scaled dot-product attention
        energy = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_size ** 0.5) # (batch_size, 1, 1)
        attention = torch.softmax(energy, dim=2) # (batch_size, 1, 1)

        out = torch.bmm(attention, values).squeeze(1) # (batch_size, embed_size)

        return self.gamma * out + x # Additive attention with skip connection element


class AttentionNN(nn.Module):
    def __init__(self, input_dim, n_units_l1=128, n_units_l2=64, dropout_l1=0.3, dropout_l2=0.3):
        super(AttentionNN, self).__init__()
        self.attention = SelfAttention(input_dim) # Attention applied first
        self.fc1 = nn.Linear(input_dim, n_units_l1) # input_dim because attention output is same dim
        self.bn1 = nn.BatchNorm1d(n_units_l1)
        self.dropout1 = nn.Dropout(dropout_l1)
        self.fc2 = nn.Linear(n_units_l1, n_units_l2)
        self.bn2 = nn.BatchNorm1d(n_units_l2)
        self.dropout2 = nn.Dropout(dropout_l2)
        self.fc3 = nn.Linear(n_units_l2, 1)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x



# --- Training function (modified for HPO reporting and clarity) ---
def train_model_hpo(model_instance, current_train_loader, current_val_loader, criterion, optimizer,
                    model_name="model", num_epochs=100, patience=10, trial=None, hpo_run=False):
    # This function is now primarily for HPO.
    # A separate function or call will handle final training.
    model_instance = model_instance.to(device)
    best_val_auc = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model_instance.train()
        running_loss = 0.0
        train_targets_epoch, train_outputs_epoch = [], []
        for inputs, targets in current_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model_instance(inputs) # Squeeze later if needed
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_targets_epoch.extend(targets.cpu().detach().numpy().flatten())
            train_outputs_epoch.extend(outputs.cpu().detach().numpy().flatten())

        # Validation phase
        model_instance.eval()
        val_loss = 0.0
        val_targets_epoch, val_outputs_epoch = [], []
        with torch.no_grad():
            for inputs, targets in current_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_instance(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_targets_epoch.extend(targets.cpu().numpy().flatten())
                val_outputs_epoch.extend(outputs.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(current_val_loader.dataset) if len(current_val_loader.dataset) > 0 else 0.0

        # Calculate validation AUC
        try:
            if len(np.unique(val_targets_epoch)) < 2 : # Check if only one class present in y_true
                 current_val_auc = 0.5 # Or some other baseline or skip if only one class
            else:
                current_val_auc = roc_auc_score(val_targets_epoch, val_outputs_epoch)
        except ValueError:
            current_val_auc = 0.0

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            no_improve_epochs = 0
            # For HPO, we don't need to save the model here, just report the best AUC
        else:
            no_improve_epochs += 1

        if hpo_run and (epoch + 1) % 10 == 0: # Print less frequently during HPO
             print(f"Trial {trial.number if trial else 'N/A'}, Epoch {epoch+1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {current_val_auc:.4f}")

        if trial: # Optuna pruning
            trial.report(current_val_auc, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()

        if no_improve_epochs >= patience:
            if hpo_run: print(f"Early stopping HPO trial at epoch {epoch+1} for {model_name}")
            break

    return best_val_auc # Return the best validation AUC for this trial


#the objective function commented out targets ROC AUC, current one targets the F1 score and adjusts the threshold
INPUT_DIM_GLOBAL = X_train_hpo_tensor.shape[1] # Should be set after data processing

#custom metric for optimizing business cost based on the 5x customer acquisition heuristic
def cost_sensitive_metric(y_true, y_pred, cost_fn=5, cost_fp=1):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost

def objective(trial, model_class_name):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    if model_class_name == "BasicNN":
        n_units_l1 = trial.suggest_int("n_units_l1", 32, 512)
        n_units_l2 = trial.suggest_int("n_units_l2", 16, 256)
        dropout_l1 = trial.suggest_float("dropout_l1", 0.1, 0.5)
        dropout_l2 = trial.suggest_float("dropout_l2", 0.1, 0.5)
        model = BasicNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, dropout_l1, dropout_l2)
    elif model_class_name == "DeepNN":
        n_units_l1 = trial.suggest_int("n_units_l1", 64, 512)
        n_units_l2 = trial.suggest_int("n_units_l2", 32, 512)
        n_units_l3 = trial.suggest_int("n_units_l3", 16, 256)
        n_units_l4 = trial.suggest_int("n_units_l4", 8, 128)
        dropout_l1 = trial.suggest_float("dropout_l1", 0, 0.6)
        dropout_l2 = trial.suggest_float("dropout_l2", 0, 0.6)
        dropout_l3 = trial.suggest_float("dropout_l3", 0, 0.5)
        dropout_l4 = trial.suggest_float("dropout_l4", 0, 0.5)
        model = DeepNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, n_units_l3, n_units_l4,
                       dropout_l1, dropout_l2, dropout_l3, dropout_l4)
    elif model_class_name == "ResNet":
        initial_dim = trial.suggest_int("initial_dim", 64, 512)
        res_block_dim = trial.suggest_int("res_block_dim", 32, initial_dim)
        dropout_res = trial.suggest_float("dropout_res", 0.1, 0.5)
        num_res_blocks = trial.suggest_int("num_res_blocks", 1, 8)
        model = ResNet(INPUT_DIM_GLOBAL, initial_dim, res_block_dim, dropout_res, num_res_blocks)
    elif model_class_name == "AttentionNN":
        n_units_l1 = trial.suggest_int("n_units_l1", 32, 512)
        n_units_l2 = trial.suggest_int("n_units_l2", 16, 512)
        dropout_l1 = trial.suggest_float("dropout_l1", 0, 0.5)
        dropout_l2 = trial.suggest_float("dropout_l2", 0, 0.5)
        model = AttentionNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, dropout_l1, dropout_l2)
    else:
        raise ValueError("Unknown model class name")

    model = model.to(device)
    num_pos = y_train_hpo_tensor.sum().item()
    num_neg = len(y_train_hpo_tensor) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=0.9, gamma=1.5).to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Train model (reusing your train_model_hpo function)
    model.train()
    #best_recall = 0.0
    #best_f1 = 0.0
    no_improve_epochs = 3
    patience = 10
    best_cost=float('inf')
    for epoch in range(50):  # fewer epochs during HPO
        model.train()
        for inputs, targets in train_hpo_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation predictions
        model.eval()
        val_targets_epoch, val_outputs_epoch = [], []
        with torch.no_grad():
            for inputs, targets in val_hpo_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = torch.sigmoid(model(inputs))
                val_outputs_epoch.extend(outputs.cpu().numpy().flatten())
                val_targets_epoch.extend(targets.cpu().numpy().flatten())
        thresholds = np.linspace(0.1, 0.9, 81)
        costs = [
            cost_sensitive_metric(
                val_targets_epoch,
                np.array(val_outputs_epoch) > t,
                cost_fn=5, cost_fp=1
            )
            for t in thresholds]
        min_cost = min(costs)
        optimal_threshold = thresholds[costs.index(min_cost)]

        if min_cost < best_cost:
            best_cost = min_cost
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        trial.report(-min_cost, epoch)  # Optuna maximizes by default; negate cost

        if trial.should_prune():
            raise optuna.TrialPruned()

        if no_improve_epochs >= patience:
            break

        '''
        #switching from f1-score to recall score
        #Recall focused
        thresholds = np.linspace(0.1, 0.9, 81)
        recall_scores = [recall_score(val_targets_epoch, np.array(val_outputs_epoch) > t, zero_division=0) for t in thresholds]
        max_recall = max(recall_scores)
        optimal_threshold = thresholds[recall_scores.index(max_recall)]
        if max_recall > best_recall:
            best_recall = max_recall
            no_improve_epochs = 0
        else:
            no_improve_epochs += 0.5

        trial.report(max_recall, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if no_improve_epochs >= patience:
            break
        '''
        '''
        # Calculate F1 score
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [f1_score(val_targets_epoch, np.array(val_outputs_epoch) > t, zero_division=0) for t in thresholds]
        max_f1 = max(f1_scores)
        optimal_threshold = thresholds[f1_scores.index(max_f1)]

        if max_f1 > best_f1:
            best_f1 = max_f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        trial.report(max_f1, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if no_improve_epochs >= patience:
            break
        '''
    # Save optimal threshold as a trial user attribute for later retrieval
    trial.set_user_attr("optimal_threshold", optimal_threshold)

    return -best_cost  # F1 score as the objective

'''
def objective(trial, model_class_name):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    if model_class_name == "BasicNN":
        n_units_l1 = trial.suggest_int("n_units_l1", 32, 256)
        n_units_l2 = trial.suggest_int("n_units_l2", 16, 128)
        dropout_l1 = trial.suggest_float("dropout_l1", 0.1, 0.5)
        dropout_l2 = trial.suggest_float("dropout_l2", 0.1, 0.5)
        model = BasicNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, dropout_l1, dropout_l2)
    elif model_class_name == "DeepNN":
        n_units_l1 = trial.suggest_int("n_units_l1", 64, 512)
        n_units_l2 = trial.suggest_int("n_units_l2", 32, 256)
        n_units_l3 = trial.suggest_int("n_units_l3", 16, 128)
        n_units_l4 = trial.suggest_int("n_units_l4", 8, 64)
        dropout_l1 = trial.suggest_float("dropout_l1", 0.1, 0.6)
        dropout_l2 = trial.suggest_float("dropout_l2", 0.1, 0.6)
        dropout_l3 = trial.suggest_float("dropout_l3", 0.1, 0.5)
        dropout_l4 = trial.suggest_float("dropout_l4", 0.1, 0.4)
        model = DeepNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, n_units_l3, n_units_l4,
                       dropout_l1, dropout_l2, dropout_l3, dropout_l4)
    elif model_class_name == "ResNet":
        initial_dim = trial.suggest_int("initial_dim", 64, 256)
        res_block_dim = trial.suggest_int("res_block_dim", 32, initial_dim) # Ensure res_block_dim <= initial_dim
        dropout_res = trial.suggest_float("dropout_res", 0.1, 0.5)
        num_res_blocks = trial.suggest_int("num_res_blocks", 1, 4)
        model = ResNet(INPUT_DIM_GLOBAL, initial_dim=initial_dim, res_block_dim=res_block_dim,
                       dropout_res=dropout_res, num_res_blocks=num_res_blocks)
    elif model_class_name == "AttentionNN":
        # Attention layer itself is not tuned here, but the subsequent MLP layers are
        n_units_l1 = trial.suggest_int("n_units_l1", 32, 256)
        n_units_l2 = trial.suggest_int("n_units_l2", 16, 128)
        dropout_l1 = trial.suggest_float("dropout_l1", 0.1, 0.5)
        dropout_l2 = trial.suggest_float("dropout_l2", 0.1, 0.5)
        model = AttentionNN(INPUT_DIM_GLOBAL, n_units_l1, n_units_l2, dropout_l1, dropout_l2)
    else:
        raise ValueError("Unknown model class name")

    model = model.to(device)
    num_pos = y_train_hpo_tensor.sum().item()
    num_neg = len(y_train_hpo_tensor) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: # SGD
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Train using HPO loaders, for fewer epochs during HPO
    # Using train_hpo_loader and val_hpo_loader
    print(f"\nStarting Optuna Trial {trial.number} for {model_class_name}...")
    val_auc = train_model_hpo(model, train_hpo_loader, val_hpo_loader, criterion, optimizer,
                              model_name=f"{model_class_name}_trial_{trial.number}",
                              num_epochs=100, patience=15, trial=trial, hpo_run=True)
    return val_auc
'''

# --- Final Training Function (after HPO) ---
def final_train_model(model_instance, final_train_loader, final_val_loader, criterion, optimizer,
                      model_name, num_epochs=100, patience=15, model_save_path_base="model_outputs"):
    model_instance = model_instance.to(device)
    best_val_auc = 0.0
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []

    model_save_path = os.path.join(model_save_path_base, f"{model_name}_best.pth")

    print(f"\nStarting final training for {model_name}...")
    for epoch in range(num_epochs):
        model_instance.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []
        for inputs, targets in final_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model_instance(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            y_true_train.extend(targets.cpu().detach().numpy().flatten())
            y_pred_train.extend(outputs.cpu().detach().numpy().flatten())

        epoch_train_loss = running_loss / len(final_train_loader.dataset) if len(final_train_loader.dataset) > 0 else 0.0
        train_losses.append(epoch_train_loss)
        try:
            if len(np.unique(y_true_train)) < 2: train_auc = 0.5
            else: train_auc = roc_auc_score(y_true_train, y_pred_train)
        except ValueError: train_auc = 0.0
        train_aucs.append(train_auc)

        model_instance.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for inputs, targets in final_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_instance(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                y_true_val.extend(targets.cpu().numpy().flatten())
                y_pred_val.extend(outputs.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(final_val_loader.dataset) if len(final_val_loader.dataset) > 0 else 0.0
        val_losses.append(epoch_val_loss)
        try:
            if len(np.unique(y_true_val)) < 2: val_auc = 0.5
            else: val_auc = roc_auc_score(y_true_val, y_pred_val)
        except ValueError: val_auc = 0.0
        val_aucs.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve_epochs = 0
            torch.save(model_instance.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: New best model saved to {model_save_path} with Val AUC: {val_auc:.4f}")
        else:
            no_improve_epochs += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} for {model_name}.")
            break

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Over Epochs - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label='Train AUC')
    plt.plot(val_aucs, label='Validation AUC')
    plt.title(f'AUC Over Epochs - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path_base, f"{model_name}_training_history.png"))
    plt.close() # Close plot to avoid display issues in loops

    # Load the best model
    if os.path.exists(model_save_path):
        model_instance.load_state_dict(torch.load(model_save_path))
    else:
        print(f"Warning: Model path {model_save_path} not found. Using model from last epoch of training.")
    return model_instance, best_val_auc


# --- Evaluation function with misclassification logging ---
def evaluate_model_and_log_failures(model, final_test_loader, original_X_test_df, original_y_test_series,
                                    model_name, model_save_path_base="model_outputs"):
    model.eval()
    y_true_list, y_pred_list, y_scores_list = [], [], []
    misclassified_samples_log = []
    current_sample_idx = 0

    print(f"\nEvaluating {model_name} on the test set...")
    with torch.no_grad():
        for inputs, targets in final_test_loader:
            inputs, targets_batch = inputs.to(device), targets.to(device) # targets_batch is [batch_size, 1]
            outputs = model(inputs) # Model output is [batch_size, 1]
            predicted_probs = outputs.cpu().numpy().flatten()
            optimal_threshold = study.best_trial.user_attrs["optimal_threshold"]
            predicted_labels = (predicted_probs > optimal_threshold).astype(float)


            true_labels_batch = targets_batch.cpu().numpy().flatten()

            y_true_list.extend(true_labels_batch)
            y_pred_list.extend(predicted_labels)
            y_scores_list.extend(predicted_probs)

            # Log misclassified samples
            for i in range(len(true_labels_batch)):
                if int(predicted_labels[i]) != int(true_labels_batch[i]):
                    if current_sample_idx < len(original_X_test_df):
                        original_features = original_X_test_df.iloc[current_sample_idx].to_dict()
                        log_entry = {
                            'original_df_index': original_X_test_df.index[current_sample_idx],
                            'true_label': int(true_labels_batch[i]),
                            'predicted_label': int(predicted_labels[i]),
                            'predicted_score_for_churn': float(predicted_probs[i]),
                            'features': original_features
                        }
                        misclassified_samples_log.append(log_entry)
                    else:
                        print(f"Warning: Index mismatch in misclassification logging for {model_name}.")
                current_sample_idx += 1

    # Save misclassified samples
    misclassified_df = pd.DataFrame(misclassified_samples_log)
    misclassified_log_path = os.path.join(model_save_path_base, f"{model_name}_misclassified_samples.csv")
    misclassified_df.to_csv(misclassified_log_path, index=False)
    print(f"Logged {len(misclassified_df)} misclassified samples to {misclassified_log_path}")

    # Calculate metrics
    accuracy = accuracy_score(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list, output_dict=False, zero_division=0) # Get string report
    conf_matrix = confusion_matrix(y_true_list, y_pred_list)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_list, y_scores_list)
    roc_auc_val = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} (Test Set)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_save_path_base, f"{model_name}_test_roc_curve.png"))
    plt.close()

    return accuracy, roc_auc_val, report, conf_matrix, fpr, tpr


# --- Main Execution ---
N_TRIALS_OPTUNA = 50 # Number of Bayesian optimization trials
EPOCHS_FINAL_TRAIN = 600 #for the final model
PATIENCE_FINAL_TRAIN = 25 # Patience for final model training

models_to_tune = {
    "BasicNN": BasicNN,
    "DeepNN": DeepNN,
    "ResNet": ResNet,
    "AttentionNN": AttentionNN
}

results = {}
# compute pos_weight to handle class imbalance when gerating the criterion - add extra scaling factor to account for cost difference
num_pos = y_train_hpo_tensor.sum().item()
num_neg = len(y_train_hpo_tensor) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=0.75, gamma=1.5).to(device)
# Ensure INPUT_DIM_GLOBAL is set
INPUT_DIM_GLOBAL = X_train_full_tensor.shape[1]
if INPUT_DIM_GLOBAL != X_train_hpo_tensor.shape[1] or INPUT_DIM_GLOBAL != X_test_tensor.shape[1]:
    print("WARNING: Mismatch in processed feature dimensions. Check preprocessing steps.")
    # This should not happen if preprocessor is fit on X_train_full and then used to transform all splits.

# Initialize plot for all ROC curves
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')

for model_name_key, model_class_constructor in models_to_tune.items():
    print(f"\n--- Optimizing {model_name_key} ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, model_name_key), n_trials=N_TRIALS_OPTUNA, timeout=1800) # Timeout per model (e.g., 30 mins)

    best_hyperparams = study.best_params
    best_val_auc_hpo = study.best_value if study.best_value is not None else 0.0 # Handle case where no trials complete or all fail
    print(f"Best HPO Validation AUC for {model_name_key}: {best_val_auc_hpo:.4f}")
    print(f"Best Hyperparameters for {model_name_key}: {best_hyperparams}")

    # Save HPO visualization (optional)
    try:
        if optuna.visualization.is_available():
            fig_opt_history = optuna.visualization.plot_optimization_history(study)
            fig_opt_history.write_image(os.path.join("hpo_visualizations", f"{model_name_key}_optuna_history.png"))
            # Slice plot can be very large, consider if needed
            # fig_slice = optuna.visualization.plot_slice(study)
            # fig_slice.write_image(os.path.join("hpo_visualizations", f"{model_name_key}_optuna_slice.png"))
        else:
            print("Optuna visualization (plotly) not available. Skipping HPO plots.")
    except Exception as e:
        print(f"Could not generate Optuna plots for {model_name_key}: {e}")


    # Instantiate model with best hyperparameters
    final_model_args = {'input_dim': INPUT_DIM_GLOBAL}
    final_model = None # Initialize to ensure it's defined

    if model_name_key == "BasicNN":
        final_model_args.update({k: v for k, v in best_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'dropout_l1', 'dropout_l2']})
        final_model = BasicNN(**final_model_args)
    elif model_name_key == "DeepNN":
        final_model_args.update({k: v for k, v in best_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'n_units_l3', 'n_units_l4', 'dropout_l1', 'dropout_l2', 'dropout_l3', 'dropout_l4']})
        final_model = DeepNN(**final_model_args)
    elif model_name_key == "ResNet":
         final_model_args.update({
            'initial_dim': best_hyperparams.get('initial_dim', 128), # Provide default if not in best_params
            'res_block_dim': best_hyperparams.get('res_block_dim', best_hyperparams.get('initial_dim', 128)),
            'dropout_res': best_hyperparams.get('dropout_res', 0.3),
            'num_res_blocks': best_hyperparams.get('num_res_blocks', 2)
        })
         final_model = ResNet(**final_model_args)
    elif model_name_key == "AttentionNN":
        final_model_args.update({k: v for k, v in best_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'dropout_l1', 'dropout_l2']})
        final_model = AttentionNN(**final_model_args)
    else:
        raise ValueError(f"Model class {model_name_key} not configured for final instantiation.")

    # Optimizer for final model
    final_lr = best_hyperparams.get('lr', 0.001)
    final_weight_decay = best_hyperparams.get('weight_decay', 1e-5)
    optimizer_name = best_hyperparams.get('optimizer', "Adam")
    final_optimizer = None # Initialize

    if final_model: # Ensure model was instantiated
        if optimizer_name == "Adam":
            final_optimizer = optim.Adam(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
        elif optimizer_name == "RMSprop":
            final_optimizer = optim.RMSprop(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
        else: # SGD
            final_optimizer = optim.SGD(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay, momentum=0.9)

        # Train the final model
        start_time = time.time()
        trained_final_model, _ = final_train_model(
            final_model, train_full_loader, val_hpo_loader, criterion, final_optimizer,
            model_name=model_name_key, num_epochs=EPOCHS_FINAL_TRAIN, patience=PATIENCE_FINAL_TRAIN
        )
        training_time = time.time() - start_time

        # Evaluate
        accuracy, roc_auc_test, report, conf_matrix, fpr, tpr = evaluate_model_and_log_failures(
            trained_final_model, test_loader, X_test_orig, y_test_orig, model_name=model_name_key
        )

        results[model_name_key] = {
            "accuracy": accuracy, "roc_auc": roc_auc_test, "training_time_seconds": training_time,
            "best_hpo_val_auc": best_val_auc_hpo, "best_hyperparams": best_hyperparams, "fpr": fpr, "tpr": tpr
        }

        print(f"\n--- Test Results for Tuned {model_name_key} ---")
        print(f"Accuracy: {accuracy:.4f}"); print(f"ROC AUC: {roc_auc_test:.4f}")
        print(f"Training Time: {training_time:.2f} seconds"); print("Classification Report:"); print(report)
        print("Confusion Matrix:"); print(conf_matrix)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name_key} (Test AUC = {roc_auc_test:.4f})')
    else:
        print(f"Skipping final training and evaluation for {model_name_key} due to instantiation failure (likely HPO issues).")


# Finalize combined ROC plot
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Tuned Deep Learning Models (Test Set)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig(os.path.join('model_outputs', 'all_models_tuned_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# --- Model Comparison ---
print("\nMODEL COMPARISON (TUNED MODELS ON TEST SET)")
print("---------------------------------------------")

if results: # Proceed only if there are results
    comparison_data = []
    for model_name, res_dict in results.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': res_dict["accuracy"],
            'Test ROC AUC': res_dict["roc_auc"],
            'HPO Val AUC': res_dict["best_hpo_val_auc"],
            'Training Time (s)': res_dict["training_time_seconds"],
            'Best Hyperparams': str(res_dict["best_hyperparams"]) # Convert dict to string for CSV
        })

    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        print("\nModel Performance Comparison:")
        print(comparison_df.sort_values('Test ROC AUC', ascending=False))
        comparison_df.to_csv(os.path.join('model_outputs', 'tuned_model_comparison.csv'), index=False)

        # Plot comparison (Accuracy, ROC AUC, Training Time)
        plt.figure(figsize=(18, 6))
        bar_width = 0.25
        model_names_short = [name.replace("Neural Network", "NN").replace("ResNet", "ResNet") for name in comparison_df['Model']]
        index = np.arange(len(model_names_short))

        plt.subplot(1, 3, 1)
        plt.bar(index, comparison_df['Test Accuracy'], bar_width, label='Test Accuracy', color='skyblue')
        plt.bar(index + bar_width, comparison_df['Test ROC AUC'], bar_width, label='Test ROC AUC', color='salmon')
        # Handle potential None in HPO Val AUC if trials failed
        hpo_val_aucs_plot = [val if pd.notna(val) else 0 for val in comparison_df['HPO Val AUC']]
        plt.bar(index + 2*bar_width, hpo_val_aucs_plot, bar_width, label='HPO Val AUC', color='lightgreen')

        plt.xlabel('Model'); plt.ylabel('Score'); plt.title('Model Performance (Scores)')
        plt.xticks(index + bar_width, model_names_short, rotation=45, ha="right"); plt.legend(); plt.ylim(0, 1)

        plt.subplot(1, 3, 2)
        plt.bar(index, comparison_df['Training Time (s)'], bar_width * 2, label='Training Time (s)', color='gold')
        plt.xlabel('Model'); plt.ylabel('Training Time (seconds)'); plt.title('Training Time')
        plt.xticks(index, model_names_short, rotation=45, ha="right"); plt.legend()

        # Display best hyperparameters for the top model
        best_overall_model_row = comparison_df.sort_values('Test ROC AUC', ascending=False).iloc[0]
        best_overall_model_name = best_overall_model_row['Model']
        best_hyperparams_text = f"Best Model: {best_overall_model_name}\n"
        # Retrieve original dict from results for better formatting
        best_hyperparams_dict_orig = results[best_overall_model_name]['best_hyperparams']
        for k,v in best_hyperparams_dict_orig.items():
            if isinstance(v, float): best_hyperparams_text += f"{k}: {v:.4f}\n"
            else: best_hyperparams_text += f"{k}: {v}\n"
        plt.subplot(1, 3, 3)
        plt.text(0.05, 0.95, best_hyperparams_text, ha='left', va='top', fontsize=9, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="grey", lw=1))
        plt.axis('off'); plt.title(f'Best Params ({best_overall_model_name})')
        plt.tight_layout(); plt.savefig(os.path.join('model_outputs', 'tuned_model_comparison_plots.png'), dpi=300); plt.show(); plt.close()
    else:
        print("Comparison DataFrame is empty, skipping plots.")
else:
    print("No results to compare or plot.")


# --- Permutation Feature Importance (on the best tuned model) ---
print("\nFEATURE IMPORTANCE ANALYSIS (Permutation Importance)")
print("---------------------------------------------")

def permutation_importance_torch(model_instance, X_data_tensor, y_data_tensor, metric_func,
                                 feature_names_list, n_repeats=10):
    model_instance.eval()
    model_instance = model_instance.to(device)
    X_data_tensor, y_data_tensor = X_data_tensor.to(device), y_data_tensor.to(device)

    with torch.no_grad():
        baseline_outputs = model_instance(X_data_tensor)
        baseline_score = metric_func(y_data_tensor.cpu().numpy().flatten(), baseline_outputs.cpu().numpy().flatten())

    importances_all_repeats = np.zeros((X_data_tensor.shape[1], n_repeats))

    for i in range(X_data_tensor.shape[1]): # For each feature
        for n in range(n_repeats):
            X_permuted = X_data_tensor.clone()
            perm_indices = torch.randperm(X_data_tensor.size(0))
            X_permuted[:, i] = X_data_tensor[perm_indices, i]
            with torch.no_grad():
                permuted_outputs = model_instance(X_permuted)
                permuted_score = metric_func(y_data_tensor.cpu().numpy().flatten(), permuted_outputs.cpu().numpy().flatten())
            importances_all_repeats[i, n] = baseline_score - permuted_score

    importances_mean = np.mean(importances_all_repeats, axis=1)
    importances_std = np.std(importances_all_repeats, axis=1)

    importance_df = pd.DataFrame({
        'Feature': feature_names_list,
        'Importance_Mean': importances_mean,
        'Importance_Std': importances_std
    })
    return importance_df.sort_values('Importance_Mean', ascending=False)


if results and not comparison_df.empty: # Check if results and comparison_df exist and are not empty
    best_model_name_overall = comparison_df.sort_values('Test ROC AUC', ascending=False).iloc[0]['Model']
    print(f"\nCalculating Permutation Importance for the best model: {best_model_name_overall}")

    best_model_hyperparams = results[best_model_name_overall]['best_hyperparams']
    best_model_args = {'input_dim': INPUT_DIM_GLOBAL}
    best_final_model_for_importance = None # Initialize

    if best_model_name_overall == "BasicNN":
        best_model_args.update({k: v for k, v in best_model_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'dropout_l1', 'dropout_l2']})
        best_final_model_for_importance = BasicNN(**best_model_args)
    elif best_model_name_overall == "DeepNN":
        best_model_args.update({k: v for k, v in best_model_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'n_units_l3', 'n_units_l4', 'dropout_l1', 'dropout_l2', 'dropout_l3', 'dropout_l4']})
        best_final_model_for_importance = DeepNN(**best_model_args)
    elif best_model_name_overall == "ResNet":
        best_model_args.update({
            'initial_dim': best_model_hyperparams.get('initial_dim',128),
            'res_block_dim': best_model_hyperparams.get('res_block_dim', best_model_hyperparams.get('initial_dim',128)),
            'dropout_res': best_model_hyperparams.get('dropout_res',0.3),
            'num_res_blocks': best_model_hyperparams.get('num_res_blocks',2)
        })
        best_final_model_for_importance = ResNet(**best_model_args)
    elif best_model_name_overall == "AttentionNN":
        best_model_args.update({k: v for k, v in best_model_hyperparams.items() if k in ['n_units_l1', 'n_units_l2', 'dropout_l1', 'dropout_l2']})
        best_final_model_for_importance = AttentionNN(**best_model_args)
    else:
        print(f"Importance calculation not configured for {best_model_name_overall}")

    if best_final_model_for_importance:
        best_model_path = os.path.join("model_outputs", f"{best_model_name_overall}_best.pth")
        if os.path.exists(best_model_path):
            best_final_model_for_importance.load_state_dict(torch.load(best_model_path, map_location=device))
            best_final_model_for_importance = best_final_model_for_importance.to(device)

            importance_df_final = permutation_importance_torch(
                best_final_model_for_importance,
                X_test_tensor,
                y_test_tensor.squeeze(),
                roc_auc_score,
                feature_names_processed,
                n_repeats=10
            )

            print(f"\nTop 15 Feature Importances for {best_model_name_overall} (Test Set):")
            print(importance_df_final.head(15))
            importance_df_final.to_csv(os.path.join('model_outputs', f'{best_model_name_overall}_permutation_importance.csv'), index=False)

            # Plot feature importance using matplotlib.pyplot.barh
            plt.figure(figsize=(12, 8))
            data_to_plot = importance_df_final.head(15)
            plt.barh(
                data_to_plot['Feature'].iloc[::-1],
                data_to_plot['Importance_Mean'].iloc[::-1],
                xerr=data_to_plot['Importance_Std'].iloc[::-1],
                align='center',
                capsize=5
            )
            plt.xlabel('Permutation Importance (Mean Decrease in ROC AUC)')
            plt.ylabel('Feature')
            plt.title(f'Top 15 Permutation Feature Importances - {best_model_name_overall} (Test Set)')
            plt.tight_layout()
            plt.savefig(os.path.join('model_outputs', f'{best_model_name_overall}_permutation_importance.png'), dpi=300)
            plt.show()
            plt.close()
        else:
            print(f"Could not load best model for permutation importance: {best_model_path} not found.")
    else:
        print(f"Best model for permutation importance could not be instantiated for {best_model_name_overall}.")
else:
    print("No results available to determine the best model for feature importance.")

print("\n=============================================")
print("DEEP LEARNING ANALYSIS WITH HPO COMPLETE")
print("=============================================")



misclassified_logs_path = "model_outputs"
data_path = "telco_churn_data.csv"
analysis_output_folder = "misclassification_analysis"
os.makedirs(analysis_output_folder, exist_ok=True)

# Load full dataset
try:
    df_full = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"ERROR: Full dataset not found at {data_path}. Please check the path.")
    exit()

# Add County feature from Zip Code if needed
if "County" not in df_full.columns and "Zip Code" in df_full.columns:
    print("Generating 'County' feature from 'Zip Code' using uszipcode...")
    search = SearchEngine()
    def get_county_by_zip(zip_code):
        result = search.by_zipcode(zip_code)
        return result.county if result else None
    df_full["County"] = df_full["Zip Code"].apply(get_county_by_zip)

# Models to analyze
model_names = ["BasicNN", "DeepNN", "ResNet", "AttentionNN"]

# Analysis Function
def detailed_feature_analysis(model_name):
    print(f"\n--- Analyzing Misclassifications for Model: {model_name} ---")
    misclassified_file = os.path.join(misclassified_logs_path, f"{model_name}_misclassified_samples.csv")

    if not os.path.exists(misclassified_file):
        print(f"Misclassification file not found for {model_name}: {misclassified_file}")
        return

    df_misclassified = pd.read_csv(misclassified_file)
    if df_misclassified.empty or 'features' not in df_misclassified.columns:
        print(f"Skipping {model_name} due to empty or missing feature column.")
        return

    # Convert features column from string to dict
    try:
        features_dict_series = df_misclassified['features'].apply(ast.literal_eval)
    except Exception:
        try:
            features_dict_series = df_misclassified['features'].apply(eval)
        except Exception as e:
            print(f"Failed to parse 'features' column for {model_name}: {e}")
            return

    features_misclassified_expanded = pd.json_normalize(features_dict_series)
    df_misclassified_combined = pd.concat([df_misclassified[['true_label']], features_misclassified_expanded], axis=1)

    available_logged_features = features_misclassified_expanded.columns.tolist()
    categorical_features_to_analyze = []
    numeric_features_to_analyze = []

    print(f"Available features in misclassified logs for {model_name}: {available_logged_features}")

    for feature_name in available_logged_features:
        if feature_name in df_full.columns:
            if pd.api.types.is_object_dtype(df_full[feature_name]) or pd.api.types.is_bool_dtype(df_full[feature_name]):
                categorical_features_to_analyze.append(feature_name)
            elif pd.api.types.is_numeric_dtype(df_full[feature_name]):
                numeric_features_to_analyze.append(feature_name)

    print(f"Categorical features to analyze: {categorical_features_to_analyze}")
    print(f"Numeric features to analyze: {numeric_features_to_analyze}")

    for feature in categorical_features_to_analyze:
        if feature not in df_misclassified_combined.columns or feature not in df_full.columns:
            print(f"Skipping missing categorical feature '{feature}'")
            continue
        total_counts = df_full[feature].value_counts(normalize=True, dropna=False)
        misclassified_counts = df_misclassified_combined[feature].value_counts(normalize=True, dropna=False)
        compare_df = pd.DataFrame({'Full Dataset': total_counts, 'Misclassified': misclassified_counts}).fillna(0)
        if compare_df.empty:
            continue
        safe_feature = feature.replace("/", "_").replace(" ", "_")
        compare_df.plot.bar(figsize=(12, 6))
        plt.title(f"{model_name}: Feature '{feature}' Proportion Comparison")
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_folder, f"{model_name}_{safe_feature}_proportion_comparison.png"))
        plt.close()

    for feature in numeric_features_to_analyze:
        if feature not in df_misclassified_combined.columns or feature not in df_full.columns:
            print(f"Skipping missing numeric feature '{feature}'")
            continue
        full_vals = df_full[feature].dropna()
        misclassified_vals = df_misclassified_combined[feature].dropna()
        if full_vals.nunique() <= 1 or misclassified_vals.nunique() <= 1:
            continue
        safe_feature = feature.replace("/", "_").replace(" ", "_")
        plt.figure(figsize=(10, 5))
        sns.kdeplot(full_vals, color='blue', label='Full Dataset', fill=True, alpha=0.4)
        sns.kdeplot(misclassified_vals, color='red', label='Misclassified', fill=True, alpha=0.4)
        plt.title(f"{model_name}: Distribution of '{feature}'")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_folder, f"{model_name}_{safe_feature}_distribution_comparison.png"))
        plt.close()

# Run analysis
for model in model_names:
    detailed_feature_analysis(model)

print("\n Detailed misclassification analysis complete. Visuals saved to 'misclassification_analysis/'")

