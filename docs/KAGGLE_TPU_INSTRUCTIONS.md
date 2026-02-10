# How to Run EWIS on Kaggle TPU

This guide outlines the steps to run the Agentic Early Warning System on Kaggle's TPU v3-8 hardware for accelerated training.

## 1. Environment Setup

1.  **Create a New Notebook** on Kaggle.
2.  In the **Session Options** (sidebar on the right):
    *   **Accelerator**: Select `TPU VM v3-8`.
    *   **Persistence**: Recommended `Files only` or `Variables and Files`.

## 2. Data Setup

You need to get the C-MAPSS data into the notebook.
1.  In the **Data** pane (right sidebar), click **Add Data**.
2.  Search for `NASA C-MAPSS Jet Engine Systems`.
3.  Alternatively, you can upload the `data/processed` folder from your local machine as a "New Dataset" if you want to skip the feature engineering phase on Kaggle.

## 3. Code Setup (Clone Repository)

Run the following in the first cell of your Kaggle notebook to clone your code and install dependencies.

```python
# Cell 1: Setup Codebase & Dependencies
!git clone https://github.com/YOUR_USERNAME/Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures.git
%cd Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures

# Install dependencies (Kaggle TPU VM usually has torch/torch_xla pre-installed, but we install project deps)
!pip install -r requirements.txt
!pip install --upgrade seaborn mlflow
```

## 4. TPU Compatibility (Crucial)

TPUs use **XLA (Accelerated Linear Algebra)**. Standard PyTorch CUDA commands won't work directly. You need to override the `DeepLearningTrainer` to use `torch_xla`.

Run this cell **before** running the training sections of the notebook.

```python
# Cell 2: TPU Imports & Trainer Override
import sys
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Add src to path
sys.path.append(os.getcwd())

# 1. Define TPU Device
# On TPU VM, this gets the TPU device
tpu_device = xm.xla_device()
print(f"Running on TPU Device: {tpu_device}")

# 2. Patch the DeepLearningTrainer for TPU
from src.models.deep_learning import DeepLearningTrainer, RULDataset
from torch.utils.data import DataLoader

class TPUDeepLearningTrainer(DeepLearningTrainer):
    def __init__(self, model, learning_rate=0.001, device=None):
        # Force device to be the TPU device passed in
        super().__init__(model, learning_rate, device) 
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=128, early_stopping_patience=10):
        # Ensure data is handled correctly
        if len(X_train.shape) == 2:
            X_train = X_train[:, np.newaxis, :]
            if X_val is not None:
                X_val = X_val[:, np.newaxis, :]
        
        train_dataset = RULDataset(X_train, y_train)
        
        # TPU Sampling: DistributedSampler is often needed for multi-core, 
        # but for single-process TPU VM, standard Shuffle is okay usually.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            # --- TPU Specific: ParallelLoader (Optional but recommended for speed) ---
            # tracker = xm.RateTracker()
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                
                # --- TPU Specific: Optimizer Step ---
                xm.optimizer_step(self.optimizer)
                # -----------------------------------
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, batch_size)
                self.val_losses.append(val_loss)
                
                # Print using xm.master_print to avoid duplicate logs in distributed setup
                if (epoch + 1) % 10 == 0:
                    xm.master_print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    xm.master_print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

print("✅ TPU Trainer Class Defined")
```

## 5. Modified Config for Kaggle Paths

Kaggle input data is read-only. We need to point the config to `/kaggle/input` and outputs to `/kaggle/working`.

```python
# Cell 3: Config Override
from src.config import settings
from pathlib import Path

class KaggleConfig:
    # Adjust this to match the actual path of the dataset you added
    # Run (!ls /kaggle/input) to check
    RAW_DATA_DIR = Path("/kaggle/input/nasa-cmapss-jet-engine-systems") 
    
    # We copy/create processed data in working directory
    PROCESSED_DATA_DIR = Path("/kaggle/working/data/processed")
    MODELS_DIR = Path("/kaggle/working/models")
    OUTPUTS_DIR = Path("/kaggle/working/reports")
    
    # Create dirs
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Kaggle Paths Configured")
```

## 6. Execution Changes

When you reach **Section 5 (LSTM)** and **Section 6 (TCN)** in the notebook:

1.  Replace `device` with `tpu_device`.
2.  Replace `DeepLearningTrainer` with `TPUDeepLearningTrainer`.

**Example:**
```python
lstm_trainer = TPUDeepLearningTrainer(
    model=lstm_model,
    learning_rate=0.001,
    device=tpu_device  # <--- Use TPU device
)
```

## Summary Checklist

- [ ] Accelerator set to TPU VM v3-8.
- [ ] Dependencies (`torch-xla`) installed/verified.
- [ ] `TPUDeepLearningTrainer` class defined and used.
- [ ] `xm.optimizer_step(optimizer)` used instead of `optimizer.step()`.
- [ ] Paths updated to `/kaggle/input` and `/kaggle/working`.
