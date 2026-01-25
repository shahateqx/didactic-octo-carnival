import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score, accuracy_score, 
                            precision_score, recall_score, f1_score, 
                            roc_auc_score)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================================================================================
# IMPROVED CONFIGURATION
# ==================================================================================
DATA_PATH = '/content/drive/MyDrive/University/thesis/adhd_processed_data.npy'
BATCH_SIZE = 16          # Reduced for better generalization
EPOCHS = 150             # Increased with early stopping
LEARNING_RATE = 0.0005   # Lower for stability
THRESHOLD = 0.3          # Lower threshold for more connections
EXPECTED_ROIS = 100
HIDDEN_DIM = 128         # Increased capacity
HEADS = 8                # More attention heads
DROPOUT = 0.4            # Stronger regularization
WEIGHT_DECAY = 1e-3      # Stronger L2
USE_ADAPTIVE_THRESHOLD = True  # Use percentile-based threshold
PERCENTILE = 85          # Keep top 15% of connections
USE_CROSS_VALIDATION = False  # Set to True for k-fold CV
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 25
USE_DATA_AUGMENTATION = True

# ==================================================================================
# FOCAL LOSS FOR CLASS IMBALANCE
# ==================================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==================================================================================
# IMPROVED MODEL ARCHITECTURE
# ==================================================================================
class ImprovedADHDHybridGAT(nn.Module):
    def __init__(self, num_nodes, num_pheno_features, hidden_dim=128, num_classes=2, heads=8, dropout=0.4):
        super(ImprovedADHDHybridGAT, self).__init__()
        
        # --- Enhanced Graph Branch with 3 GAT layers ---
        self.gat1 = GATConv(num_nodes, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # --- Enhanced Phenotypic Branch ---
        self.pheno_mlp = nn.Sequential(
            nn.Linear(num_pheno_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout - 0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- Attention-based Fusion Mechanism ---
        fusion_dim = hidden_dim + 16
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # --- Enhanced Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout - 0.1),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, edge_index, batch, pheno_data):
        # Graph processing with residual connections
        x1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        x2 = F.elu(self.bn2(self.gat2(x1, edge_index)))
        x3 = F.elu(self.bn3(self.gat3(x2, edge_index)))
        
        # Global pooling (combine mean and max)
        x_graph_mean = global_mean_pool(x3, batch)
        x_graph_max = global_max_pool(x3, batch)
        x_graph = (x_graph_mean + x_graph_max) / 2
        
        # Phenotypic processing
        x_pheno = self.pheno_mlp(pheno_data)
        
        # Attention-based fusion
        combined = torch.cat((x_graph, x_pheno), dim=1)
        attention_weights = self.attention(combined)
        combined = combined * attention_weights
        
        # Classification
        out = self.classifier(combined)
        return out

# ==================================================================================
# DATA AUGMENTATION
# ==================================================================================
def augment_brain_graph(matrix, noise_level=0.03, dropout_prob=0.05):
    """Augment brain connectivity matrices with noise and dropout"""
    augmented = matrix.copy()
    
    # Add small Gaussian noise
    noise = np.random.normal(0, noise_level, matrix.shape)
    augmented += noise
    
    # Random edge dropout
    mask = np.random.random(matrix.shape) > dropout_prob
    augmented *= mask
    
    # Ensure symmetry
    augmented = (augmented + augmented.T) / 2
    
    return augmented

# ==================================================================================
# IMPROVED GRAPH CONSTRUCTION
# ==================================================================================
def create_adaptive_graph(matrix, percentile=85):
    """Create graph using adaptive percentile-based threshold"""
    threshold = np.percentile(np.abs(matrix), percentile)
    adjacency = np.abs(matrix) > threshold
    np.fill_diagonal(adjacency, 0)
    return adjacency

def create_knn_graph(matrix, k=15):
    """Create k-nearest neighbor graph"""
    adjacency = np.zeros_like(matrix, dtype=bool)
    for i in range(len(matrix)):
        # Get top-k connections for each node (excluding self)
        abs_corr = np.abs(matrix[i])
        abs_corr[i] = -np.inf  # Exclude self-connection
        top_k_indices = np.argsort(abs_corr)[-k:]
        adjacency[i, top_k_indices] = True
        adjacency[top_k_indices, i] = True
    np.fill_diagonal(adjacency, False)
    return adjacency

# ==================================================================================
# VISUALIZATION FUNCTIONS
# ==================================================================================
def plot_learning_curves(train_losses, val_accuracies, val_losses, epochs):
    """Plot training/validation loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(range(1, epochs + 1), val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(range(1, epochs + 1), val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve and calculate AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    return avg_precision

def plot_feature_importance(model, feature_names=None):
    """Plot feature importance from model weights"""
    classifier_weights = []
    for layer in model.classifier:
        if hasattr(layer, 'weight'):
            weights = layer.weight.detach().cpu().numpy()
            classifier_weights.append(weights)
    
    if len(classifier_weights) > 0:
        weights = np.abs(classifier_weights[0]).mean(axis=0)
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(weights))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': weights
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][-20:], importance_df['Importance'][-20:])
        plt.xlabel('Average Absolute Weight', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_dimensionality_reduction(X, y, method='tsne', title='Dimensionality Reduction'):
    """Plot t-SNE or PCA visualization"""
    plt.figure(figsize=(10, 8))
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_reduced = reducer.fit_transform(X)
        method_name = 't-SNE'
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method_name = 'PCA'
        explained_var = reducer.explained_variance_ratio_.sum() * 100
        title += f' ({explained_var:.1f}% variance explained)'
    
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                         c=y, cmap='viridis', alpha=0.7,
                         edgecolors='k', linewidth=0.5, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{method}_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dataset_statistics_table(matrices, phenos, labels):
    """Create and display dataset statistics table"""
    stats = {
        'Statistic': [
            'Total Subjects',
            'ADHD Patients',
            'Healthy Controls',
            'Matrix Shape',
            'Phenotypic Features',
            'Missing Values',
            'Class Balance Ratio'
        ],
        'Value': [
            f"{len(labels)}",
            f"{np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)",
            f"{len(labels)-np.sum(labels)} ({(len(labels)-np.sum(labels))/len(labels)*100:.1f}%)",
            f"{matrices.shape[1]}x{matrices.shape[2]} (ROIs)",
            f"{phenos.shape[1]}",
            f"{np.isnan(matrices).sum() + np.isnan(phenos).sum()}",
            f"{(len(labels)-np.sum(labels))/np.sum(labels):.2f}:1 (Healthy:ADHD)"
        ]
    }
    df_stats = pd.DataFrame(stats)
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(tabulate(df_stats, headers='keys', tablefmt='grid', showindex=False))
    print("="*60)
    return df_stats

def create_performance_metrics_table(y_true, y_pred, y_pred_proba):
    """Create comprehensive performance metrics table"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
    
    tn = (y_pred[y_true == 0] == 0).sum()
    fp = (y_pred[y_true == 0] == 1).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall (Sensitivity)',
            'Specificity',
            'F1-Score',
            'ROC-AUC',
            'Average Precision',
            'Balanced Accuracy'
        ],
        'Value': [
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{specificity:.4f}",
            f"{f1:.4f}",
            f"{roc_auc:.4f}",
            f"{avg_precision:.4f}",
            f"{(recall + specificity)/2:.4f}"
        ],
        'Interpretation': [
            'Overall correctness',
            'Positive predictive value',
            'True positive rate',
            'True negative rate',
            'Harmonic mean of precision and recall',
            'Area under ROC curve',
            'Area under Precision-Recall curve',
            'Average of sensitivity and specificity'
        ]
    }
    df_metrics = pd.DataFrame(metrics)
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(tabulate(df_metrics, headers='keys', tablefmt='grid', showindex=False))
    print("="*80)
    return df_metrics

def create_hyperparameter_table():
    """Create hyperparameter configuration table"""
    hyperparams = {
        'Hyperparameter': [
            'Batch Size',
            'Epochs',
            'Learning Rate',
            'Hidden Dimension',
            'GAT Heads',
            'Dropout Rate',
            'Weight Decay',
            'Graph Construction',
            'Threshold/Percentile',
            'Expected ROIs',
            'Optimizer',
            'Loss Function',
            'LR Scheduler',
            'Early Stopping'
        ],
        'Value': [
            f"{BATCH_SIZE}",
            f"{EPOCHS}",
            f"{LEARNING_RATE}",
            f"{HIDDEN_DIM}",
            f"{HEADS}",
            f"{DROPOUT}",
            f"{WEIGHT_DECAY}",
            f"{'Adaptive Percentile' if USE_ADAPTIVE_THRESHOLD else 'Fixed Threshold'}",
            f"{PERCENTILE}th percentile" if USE_ADAPTIVE_THRESHOLD else f"{THRESHOLD}",
            f"{EXPECTED_ROIS}",
            f"Adam",
            f"FocalLoss (α=0.6, γ=2.0)",
            f"ReduceLROnPlateau",
            f"Patience={EARLY_STOPPING_PATIENCE}"
        ],
        'Description': [
            'Number of samples per batch',
            'Maximum training iterations',
            'Initial step size for optimizer',
            'Hidden layer dimension in GAT',
            'Number of attention heads',
            'Dropout probability',
            'L2 regularization strength',
            'Method for creating graph edges',
            'Connection selection criterion',
            'Expected number of brain regions',
            'Optimization algorithm',
            'Classification loss function',
            'Adaptive learning rate reduction',
            'Stop if no improvement'
        ]
    }
    df_hyperparams = pd.DataFrame(hyperparams)
    print("\n" + "="*80)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*80)
    print(tabulate(df_hyperparams, headers='keys', tablefmt='grid', showindex=False))
    print("="*80)
    return df_hyperparams

def create_confusion_matrix_table(y_true, y_pred):
    """Create formatted confusion matrix table"""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                        index=['Actual Healthy', 'Actual ADHD'],
                        columns=['Predicted Healthy', 'Predicted ADHD'])
    
    cm_percent = cm / cm.sum() * 100
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print("\nCounts:")
    print(tabulate(cm_df, headers='keys', tablefmt='grid'))
    print("\nPercentages (%):")
    print(tabulate(pd.DataFrame(cm_percent,
                               index=['Actual Healthy', 'Actual ADHD'],
                               columns=['Predicted Healthy', 'Predicted ADHD']),
                  headers='keys', tablefmt='grid', floatfmt=".1f"))
    
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives', 'Total'],
        'Count': [tn, fp, fn, tp, tn+fp+fn+tp],
        'Percentage': [f"{tn/cm.sum()*100:.1f}%", f"{fp/cm.sum()*100:.1f}%",
                      f"{fn/cm.sum()*100:.1f}%", f"{tp/cm.sum()*100:.1f}%", '100%']
    }
    print("\nDetailed Breakdown:")
    print(tabulate(pd.DataFrame(metrics), headers='keys', tablefmt='grid', showindex=False))
    print("="*50)
    return cm_df

# ==================================================================================
# DATA LOADING & FILTERING
# ==================================================================================
print("Loading data...")
raw_data = np.load(DATA_PATH, allow_pickle=True)

valid_matrices = []
valid_phenos = []
valid_labels = []
skipped_count = 0

print("Filtering subjects with partial brain coverage...")
for d in raw_data:
    mat = d['matrix']
    if mat.shape == (EXPECTED_ROIS, EXPECTED_ROIS):
        valid_matrices.append(mat)
        valid_phenos.append(d['pheno'])
        valid_labels.append(d['label'])
    else:
        skipped_count += 1

print(f"Skipped {skipped_count} subjects due to incorrect matrix size.")
print(f"Retained {len(valid_matrices)} valid subjects.")

matrices = np.array(valid_matrices)
phenos = np.array(valid_phenos)
labels = np.array(valid_labels)

print(f"Class Distribution: {np.bincount(labels)}")

# ==================================================================================
# GENERATE INITIAL VISUALIZATIONS AND TABLES
# ==================================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS AND TABLES")
print("="*80)

df_stats = create_dataset_statistics_table(matrices, phenos, labels)
df_hyperparams = create_hyperparameter_table()

n_subjects, n_nodes, _ = matrices.shape
flat_matrices = matrices.reshape(n_subjects, -1)

print("\nGenerating Dimensionality Reduction Plots...")
sample_size = min(200, len(flat_matrices))
sample_idx = np.random.choice(len(flat_matrices), sample_size, replace=False)

plot_dimensionality_reduction(
    flat_matrices[sample_idx],
    labels[sample_idx],
    method='tsne',
    title='t-SNE Visualization of Brain Connectivity Features'
)

plot_dimensionality_reduction(
    flat_matrices[sample_idx],
    labels[sample_idx],
    method='pca',
    title='PCA Visualization of Brain Connectivity Features'
)

# ==================================================================================
# PREPARE DATA FOR TRAINING
# ==================================================================================
X_combined = np.hstack([flat_matrices, phenos])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_combined, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_raw, y_train)
print(f"New Training Size: {len(y_train_res)}")
print(f"New Class Distribution: {np.bincount(y_train_res)}")

# ==================================================================================
# IMPROVED GRAPH CONSTRUCTION
# ==================================================================================
def create_graph_list(X_flat, y, is_training=False):
    graph_list = []
    matrix_size = n_nodes * n_nodes
    
    for i in range(len(X_flat)):
        mat_flat = X_flat[i, :matrix_size]
        pheno = X_flat[i, matrix_size:]
        matrix = mat_flat.reshape(n_nodes, n_nodes)
        
        # Apply data augmentation if training
        if is_training and USE_DATA_AUGMENTATION:
            matrix = augment_brain_graph(matrix)
        
        # Create graph using adaptive threshold
        if USE_ADAPTIVE_THRESHOLD:
            adjacency = create_adaptive_graph(matrix, percentile=PERCENTILE)
        else:
            adjacency = np.abs(matrix) > THRESHOLD
            np.fill_diagonal(adjacency, 0)
        
        edge_index = np.array(np.where(adjacency))
        
        # Handle empty graphs
        if edge_index.shape[1] == 0:
            # Create minimal self-connections if no edges
            edge_index = np.array([[0], [0]])
        
        x_tensor = torch.tensor(matrix, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        pheno_tensor = torch.tensor(pheno, dtype=torch.float).unsqueeze(0)
        y_tensor = torch.tensor(y[i], dtype=torch.long)
        
        data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor, pheno=pheno_tensor)
        graph_list.append(data)
    
    return graph_list

print("Constructing graphs...")
train_graphs = create_graph_list(X_train_res, y_train_res, is_training=True)
test_graphs = create_graph_list(X_test_raw, y_test, is_training=False)

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

# ==================================================================================
# TRAINING FUNCTION
# ==================================================================================
def train_model(train_loader, test_loader, device):
    """Train the improved model with all enhancements"""
    
    model = ImprovedADHDHybridGAT(
        num_nodes=n_nodes, 
        num_pheno_features=3,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Use Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.6, gamma=2.0)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    print("\nStarting Training with Enhanced Model...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.pheno.reshape(-1, 3))
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.pheno.reshape(-1, 3))
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                
                preds = out.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        acc = correct / total
        val_accuracies.append(acc)
        
        # Learning rate scheduling
        scheduler.step(acc)
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), 'best_adhd_model_improved.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation accuracy: {best_acc:.4f}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies, best_acc

# ==================================================================================
# TRAINING
# ==================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*80}")
print(f"Training on: {device}")
print(f"{'='*80}")

model, train_losses, val_losses, val_accuracies, best_acc = train_model(
    train_loader, test_loader, device
)

# ==================================================================================
# GENERATE LEARNING CURVES
# ==================================================================================
print("\nGenerating Learning Curves...")
plot_learning_curves(train_losses, val_accuracies, val_losses, len(train_losses))

# ==================================================================================
# FINAL EVALUATION
# ==================================================================================
print("\n" + "="*80)
print("FINAL EVALUATION (Best Model)")
print("="*80)

model.load_state_dict(torch.load('best_adhd_model_improved.pth'))
model.eval()

y_true = []
y_pred = []
y_pred_proba = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.pheno.reshape(-1, 3))
        probabilities = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_pred_proba.extend(probabilities.cpu().numpy())

y_pred_proba = np.array(y_pred_proba)

# ==================================================================================
# GENERATE ALL FINAL VISUALIZATIONS AND TABLES
# ==================================================================================

df_metrics = create_performance_metrics_table(y_true, y_pred, y_pred_proba)
cm_df = create_confusion_matrix_table(y_true, y_pred)

print("\nGenerating ROC Curve...")
roc_auc = plot_roc_curve(y_true, y_pred_proba)

print("\nGenerating Precision-Recall Curve...")
avg_precision = plot_precision_recall_curve(y_true, y_pred_proba)

print("\nGenerating Feature Importance Plot...")
feature_names = [f'Graph_Feature_{i}' for i in range(HIDDEN_DIM)] + [f'Pheno_Feature_{i}' for i in range(16)]
plot_feature_importance(model, feature_names)

# ==================================================================================
# SUMMARY REPORT
# ==================================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)
print(f"Configuration: {'Adaptive Percentile' if USE_ADAPTIVE_THRESHOLD else 'Fixed Threshold'}")
print(f"Graph Construction: {PERCENTILE}th percentile" if USE_ADAPTIVE_THRESHOLD else f"Threshold: {THRESHOLD}")
print(f"Data Augmentation: {'Enabled' if USE_DATA_AUGMENTATION else 'Disabled'}")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"Final Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
print(f"Total Training Graphs: {len(train_graphs)}")
print(f"Total Test Graphs: {len(test_graphs)}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training Epochs Completed: {len(train_losses)}")
print("="*80)

# ==================================================================================
# SAVE ALL OUTPUTS
# ==================================================================================
df_stats.to_csv('dataset_statistics_improved.csv', index=False)
df_metrics.to_csv('performance_metrics_improved.csv', index=False)
df_hyperparams.to_csv('hyperparameters_improved.csv', index=False)
cm_df.to_csv('confusion_matrix_improved.csv', index=True)

# Save training history
history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})
history_df.to_csv('training_history.csv', index=False)

print("\n" + "="*80)
print("ALL FILES SAVED SUCCESSFULLY")
print("="*80)
print("\nVisualization files:")
print("- learning_curves.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")
print("- feature_importance.png")
print("- tsne_visualization.png")
print("- pca_visualization.png")

print("\nData files:")
print("- dataset_statistics_improved.csv")
print("- performance_metrics_improved.csv")
print("- hyperparameters_improved.csv")
print("- confusion_matrix_improved.csv")
print("- training_history.csv")

print("\nModel files:")
print("- best_adhd_model_improved.pth")

print("\n" + "="*80)
print("🎉 IMPROVED PIPELINE COMPLETE! 🎉")
print("="*80)
