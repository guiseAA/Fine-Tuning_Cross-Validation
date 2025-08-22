import json
import warnings
import os
import time
from transformers import Trainer, TrainingArguments
from dataset import CodeAffinityDataset
from model import load_model_and_tokenizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from google.colab import drive

# Caminhos base no Google Drive (DEVE ser configurado antes de rodar este código)
results_base_path = "/content/drive/MyDrive/code_affinity/results"
os.makedirs(results_base_path, exist_ok=True)

# Arquivo para salvar o fold atual
fold_file_path = os.path.join(results_base_path, "current_fold.txt")

# Arquivo de log para tempos de execução
log_file_path = os.path.join(results_base_path, "training_time_log.txt")

# Função para escrever logs
def log_training_time(fold, start_time, end_time):
    elapsed_time = end_time - start_time
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Fold {fold + 1} - Tempo de treino: {elapsed_time:.2f} segundos\n")

# Definir métricas para avaliação
def compute_metrics(p):
    preds = p.predictions.squeeze()
    labels = p.label_ids
    preds_class = (preds >= 0.5).astype(int)

    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    precision = recall = f1 = auc = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        try:
            precision = precision_score(labels, preds_class)
            recall = recall_score(labels, preds_class)
            f1 = f1_score(labels, preds_class)
            auc = roc_auc_score(labels, preds)
        except Exception as e:
            print("Erro ao calcular métricas de classificação:", e)

    return {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Função para treinar o modelo com validação cruzada
def train_model(dataset, num_splits=5):
    from sklearn.model_selection import KFold

    Method_Code = dataset['SourceCodeMethod'].tolist()
    Snipped_Code = dataset['fragment'].tolist()
    affinities = dataset['y'].tolist()

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    folds = list(kf.split(Method_Code))

    # Verificar fold salvo (se existir)
    start_fold = 0
    if os.path.exists(fold_file_path):
        with open(fold_file_path, "r") as f:
            start_fold = int(f.read().strip())
            print(f"Retomando do fold {start_fold + 1}...")

    for fold in range(start_fold, num_splits):
        train_idx, val_idx = folds[fold]

        tokenizer, model = load_model_and_tokenizer()

        print(f"Treinando Fold {fold + 1}...")

        train_Method_Code = [Method_Code[i] for i in train_idx]
        val_Method_Code = [Method_Code[i] for i in val_idx]
        train_snippets = [Snipped_Code[i] for i in train_idx]
        val_snippets = [Snipped_Code[i] for i in val_idx]
        train_affinities = [affinities[i] for i in train_idx]
        val_affinities = [affinities[i] for i in val_idx]

        train_dataset = CodeAffinityDataset(train_Method_Code, train_snippets, train_affinities, tokenizer)
        val_dataset = CodeAffinityDataset(val_Method_Code, val_snippets, val_affinities, tokenizer)

        fold_output_dir = os.path.join(results_base_path, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=5,
            logging_dir=os.path.join(fold_output_dir, "logs"),
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # Verifica se há checkpoint
        last_checkpoint = None
        checkpoints = [
            os.path.join(fold_output_dir, d)
            for d in os.listdir(fold_output_dir)
            if d.startswith("checkpoint") and os.path.isdir(os.path.join(fold_output_dir, d))
        ]
        if checkpoints:
            last_checkpoint = sorted(checkpoints)[-1]
            print(f"Checkpoint encontrado: {last_checkpoint}")
        else:
            print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")

        # Iniciar contagem de tempo antes do treinamento
        start_time = time.time()

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Capturar o tempo após o treinamento
        end_time = time.time()

        # Registrar o tempo de execução do fold
        log_training_time(fold, start_time, end_time)

        model_save_path = os.path.join(fold_output_dir, "model_trained")
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        metrics = trainer.evaluate()
        with open(os.path.join(fold_output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Salva o fold atual
        with open(fold_file_path, "w") as f:
            f.write(str(fold + 1))