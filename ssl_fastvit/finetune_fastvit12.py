import os
import random
import torch
import torch.nn as nn
import time
import torch.cuda.amp as amp
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from medmnist import INFO
from medmnist.dataset import OCTMNIST
from thop import profile
from torch.utils.tensorboard import SummaryWriter

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    FINE_TUNE_EPOCHS = 1
    K_FOLDS = 10
    SUBSET_FRACTION = 0.05129415
    RANDOM_SEED = 42
    TEST_SUBSET_SIZE = 500
    CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5"
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.5
    MODEL_SAVE_PATH = "./ssp_retinaloct_iccv2025/ssl_fastvit_t12/best_model.pth"

os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_VISIBLE_DEVICES

# Load Pretrained Backbone
from ssl_fastvit12 import ViTBackbone

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# Load OCTMNIST Dataset
info = INFO["octmnist"]
num_classes = len(info["label"])
labeled_dataset = OCTMNIST(split="train", transform=transform, download=True)
test_dataset = OCTMNIST(split="test", transform=transform, download=True)

# Select a Subset of the Dataset
def get_subset(dataset, fraction, seed):
    random.seed(seed)
    total_samples = len(dataset)
    subset_size = int(total_samples * fraction)
    indices = random.sample(range(total_samples), subset_size)
    return Subset(dataset, indices)

# Apply Subsetting
small_labeled_dataset = get_subset(labeled_dataset, Config.SUBSET_FRACTION, Config.RANDOM_SEED)
test_subset_indices = random.sample(range(len(test_dataset)), Config.TEST_SUBSET_SIZE)
test_subset = Subset(test_dataset, test_subset_indices)
test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Model Definition
class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViTBackbone()
        self.fc = nn.Sequential(
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

# Compute FLOPs (GFLOPs) and Track GPU Memory Usage
def compute_flops_and_memory(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops = flops / 1e9  
    return flops, params

def track_gpu_memory():
    allocated = torch.cuda.memory_allocated(Config.DEVICE) / 1e9  
    reserved = torch.cuda.memory_reserved(Config.DEVICE) / 1e9  
    return allocated, reserved

# Training Function with Time Tracking
def fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=Config.FINE_TUNE_EPOCHS, patience=3):
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    # Track Training Time
    total_train_time = 0
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, labels = x.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze().long()

            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        total_train_time += epoch_time

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(Config.DEVICE), labels.to(Config.DEVICE)
                labels = labels.squeeze().long()

                outputs = model(x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Val Loss", val_loss, epoch)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.4f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state_dict)
    writer.close()
    return total_train_time

# AUC Computation and Curve Plotting
def compute_auc_and_plot_fold(model, val_loader, classes, fold):
    val_labels, val_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze().long()

            outputs = model(x)
            val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    one_hot_labels = np.eye(len(classes))[val_labels]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], val_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    mean_auc = np.mean(list(roc_auc.values()))

    return fpr, tpr, roc_auc, mean_auc, val_labels, val_probs

# Evaluation on Test Data (with Confusion Matrix Visualization)
def evaluate_test_data(model, test_loader, classes):
    test_labels, test_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze().long()

            outputs = model(x)
            test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    one_hot_labels = np.eye(len(classes))[test_labels]

    # Compute confusion matrix
    predictions = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(len(classes))])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix")
    plt.savefig("./ssp_retinaloct_iccv2025/ssl_fastvit_t12/result/confusion_matrix.png")
    plt.show()

    # Classification report
    report = classification_report(test_labels, predictions, target_names=[str(i) for i in range(len(classes))])
    print(f"\nClassification Report:\n{report}")

# Stratified K-Fold Cross-Validation with Best Model Selection
skf = StratifiedKFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)
fpr_dict, tpr_dict, auc_dict = {}, {}, {}
all_fprs, all_tprs, all_auc = [], [], []

best_auc = 0.0
best_model = None
total_gpu_memory = 0.0
inference_times = []  
gpu_memory_allocated = []
gpu_memory_reserved = []
throughputs = []

scaler = amp.GradScaler()  # For mixed precision

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(small_labeled_dataset)),
                                                      [small_labeled_dataset.dataset.labels[i] for i in small_labeled_dataset.indices])):
    print(f"\nFold {fold + 1}/{Config.K_FOLDS}")
    train_subset = Subset(small_labeled_dataset, train_idx)
    val_subset = Subset(small_labeled_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    train_targets = np.array([small_labeled_dataset.dataset.labels[i] for i in train_idx]).squeeze()
    class_weights = compute_class_weight("balanced", classes=np.unique(train_targets), y=train_targets)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE))

    model = FineTunedModel(num_classes=num_classes).to(Config.DEVICE)
    model.backbone.load_state_dict(torch.load("./ssp_retinaloct_iccv2025/ssl_fastvit_t12/octmnist_ssl_fastvit_t12_model.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Track GPU Memory Usage 
    allocated, reserved = track_gpu_memory()
    gpu_memory_allocated.append(allocated)
    gpu_memory_reserved.append(reserved)

    # Training
    total_train_time = fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler)

    # Measure Inference Latency and Throughput for each fold
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        sample_input = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
        _ = model(sample_input)
    inference_time = time.time() - start_time
    throughput = 1 / inference_time  
    inference_times.append(inference_time)  
    throughputs.append(throughput)

    print(f"GPU Memory Allocated (GB): {allocated:.4f}, GPU Memory Reserved (GB): {reserved:.4f}")
    print(f"Inference Latency for One Sample (Fold {fold + 1}): {inference_time:.4f}s")
    print(f"Throughput for Fold {fold + 1}: {throughput:.4f} samples/second")

    fpr_dict[fold], tpr_dict[fold], auc_dict[fold], mean_auc, val_labels, val_probs = compute_auc_and_plot_fold(
        model, val_loader, [str(i) for i in range(num_classes)], fold + 1
    )

    # Store fold-specific AUC and comparison for best fold model
    all_fprs.append(fpr_dict[fold])
    all_tprs.append(tpr_dict[fold])
    all_auc.append(mean_auc)

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model

# Compute mean metrics across folds
mean_gpu_memory_allocated = np.mean(gpu_memory_allocated)
mean_gpu_memory_reserved = np.mean(gpu_memory_reserved)
mean_throughput = np.mean(throughputs)

# Calculate Mean Inference Time after all folds
mean_inference_time = np.mean(inference_times)
print(f"\nMean GPU Memory Allocated across folds: {mean_gpu_memory_allocated:.4f} GB")
print(f"Mean GPU Memory Reserved across folds: {mean_gpu_memory_reserved:.4f} GB")
print(f"Mean Throughput across folds: {mean_throughput:.4f} samples/sec")
print(f"\nMean Inference Time across folds: {mean_inference_time:.4f}s")

# Evaluate on Test Data using the best model
print("\nEvaluating on Test Data using the Best Model:")
evaluate_test_data(best_model, test_loader, [str(i) for i in range(num_classes)])

# Print average AUC for all folds
print(f"\nAverage AUC across folds: {np.mean(all_auc):.4f}")

# Final AUC and ROC plots across folds
plt.figure(figsize=(10, 8))
for fold in range(Config.K_FOLDS):
    plt.plot(fpr_dict[fold][0], tpr_dict[fold][0], lw=2, label=f"Fold {fold + 1} (AUC = {auc_dict[fold][0]:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) for each Fold")
plt.legend(loc="lower right")
plt.savefig("./ssp_retinaloct_iccv2025/ssl_fastvit_t12/result/roc_curve.png")
plt.show()