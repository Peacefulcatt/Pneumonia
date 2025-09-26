# Pneumonia Detection from Chest X-rays

This project builds a deep learning pipeline to classify chest X-ray images into **Normal** vs **Pneumonia** using transfer learning (EfficientNet). It includes preprocessing, training, fine-tuning, evaluation, and model explainability with Grad-CAM.

---

## Project Structure

### 1. Setup
- Import libraries (TensorFlow, NumPy, Matplotlib, etc.).
- Configure environment (GPU, seeds, mixed precision).

### 2. Paths
- Define dataset and output directories.
- Assumes Kaggle or local folder structure: `train/`, `val/`, `test/` with subfolders `NORMAL` and `PNEUMONIA`.

### 3. Quick EDA
- Preview dataset distribution and sample images.
- Helps verify dataset integrity and check for imbalance.

### 4. Data Generators
- Use `ImageDataGenerator` or `tf.data` pipeline for preprocessing.
- Apply augmentation (rescale, rotation, shift, flip).
- Create train, validation, and test iterators.

### 5. Model (Transfer Learning)
- Load pretrained EfficientNetB0.
- Freeze backbone at first, add custom classification head (Dense + Dropout).
- Compile with `binary_crossentropy` and Adam optimizer.

### 6. Callbacks & Training
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint for stability.
- Train initial frozen model.

### 7. Fine-Tuning (Optional)
- Unfreeze top convolutional blocks with low learning rate.
- Allows adaptation to chest X-ray domain.

### 8. Evaluation
- Evaluate test set with accuracy, confusion matrix, ROC-AUC, PR curves.
- Emphasize recall (sensitivity), critical in pneumonia screening.

### 9. Explainability (Grad-CAM)
- Generate Grad-CAM heatmaps over chest X-rays.
- Highlights lung regions influencing the model’s decision.

### 10. Save Model
- Save best weights and final model for reuse.
- Store class indices and training history.

### 11. Visualization
- Plot training/validation loss and accuracy curves.
- Identify overfitting or underfitting patterns.

---

## How to Run

### On Kaggle
1. Upload the notebook (`pneumonia-model_with_explanations.ipynb`) to your Kaggle environment.
2. Place dataset in `/kaggle/input/chest_xray/` or update the path in **Cell 2 (Paths)**.
3. Run all cells sequentially.

### Locally
1. Install dependencies:
   ```bash
   pip install tensorflow matplotlib scikit-learn
### Kaggle 
www.kaggle.com/code/peacefulcat/pneumonia-model
