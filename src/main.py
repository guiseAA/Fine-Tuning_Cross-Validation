# main.py
import pandas as pd
from trainer import train_model
from analyze_folds import aggregate_fold_metrics, select_best_model


# dataset
dataset = pd.read_csv('data/dataset_Training_.csv')

# Treinar o Modelo
train_model(dataset)

# Métricas
results_dir = "/content/drive/MyDrive/code_affinity/results"
num_folds = 5
metric_to_use = "eval_mse"       # ou "r2", "accuracy"
comparison_type = "lower"   # "higher" para r2/accuracy

df_metrics = aggregate_fold_metrics(results_dir, num_folds)
if df_metrics is not None:
    select_best_model(results_dir, metric_to_use, comparison_type)
