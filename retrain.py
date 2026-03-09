"""Quick retrain script with enhanced features."""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from data.dataset_builder import create_demo_dataset
from preprocessing.text_pipeline import preprocess_text
from preprocessing.feature_engineering import build_feature_matrix
from training.train_pipeline import run_benchmark

# Build dataset with enhanced features
print("Building demo dataset...")
df = create_demo_dataset()
print(f'Dataset: {len(df)} samples')

# Preprocess text
print("Preprocessing headlines...")
df["clean_text"] = df["title"].apply(preprocess_text)

# Build feature matrix with sentence embeddings + VADER
print("Building feature matrix (this may take ~30s for embeddings)...")
X, y = build_feature_matrix(df, text_col="clean_text", use_sentence_embeddings=True)
print(f'Features: {X.shape[1]} dimensions')

# Train all models
print("\nTraining models...")
results = run_benchmark(X, y, tune=True)

print('\nTraining Results:')
print(results.to_string(index=False))
