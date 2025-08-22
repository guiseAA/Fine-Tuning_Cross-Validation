import os
import json
import pandas as pd
import shutil

def aggregate_fold_metrics(results_dir="./results", num_folds=5, output_file="cross_validation_results.csv"):
    all_metrics = []

    for fold in range(1, num_folds + 1):
        metrics_path = os.path.join(results_dir, f"fold_{fold}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metrics['fold'] = fold
                all_metrics.append(metrics)
        else:
            print(f"[Aviso] Métricas não encontradas para fold {fold}.")

    if not all_metrics:
        print("Nenhuma métrica encontrada. Abortando.")
        return None

    df = pd.DataFrame(all_metrics)
    df.set_index('fold', inplace=True)

    # Calcular média e desvio padrão
    df.loc['mean'] = df.mean(numeric_only=True)
    df.loc['std'] = df.std(numeric_only=True)

    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path)
    print(f"✅ Tabela de métricas salva em: {output_path}")
    return df

def select_best_model(results_dir="./results", metric="mse", better="lower", model_dirname="model_trained", output_dirname="best_model"):
    best_value = float('inf') if better == "lower" else float('-inf')
    best_fold = None

    for fold in range(1, 6):
        metrics_path = os.path.join(results_dir, f"fold_{fold}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                value = metrics.get(metric)
                if value is not None:
                    if (better == "lower" and value < best_value) or (better == "higher" and value > best_value):
                        best_value = value
                        best_fold = fold

    if best_fold is not None:
        print(f"🏆 Melhor modelo: Fold {best_fold} com {metric} = {best_value:.4f}")
        source = os.path.join(results_dir, f"fold_{best_fold}", model_dirname)
        destination = os.path.join(results_dir, output_dirname)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"✅ Modelo copiado para: {destination}")
    else:
        print("❌ Não foi possível determinar o melhor modelo.")

