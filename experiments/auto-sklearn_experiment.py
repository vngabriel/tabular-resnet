import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from auto_sklearn2 import AutoSklearnClassifier

from data.data import DataProcessor
from evaluation.metrics import calculate_all_metrics

# Configuration
SEED = 123
CACHE_DIR = "./data/openml_cache"
RESULTS_DIR = "./results/autosklearn2"
processor = DataProcessor(seed=SEED, cache_dir=CACHE_DIR)

# Find all processed PKL files
pkl_files = sorted(list(Path(CACHE_DIR).glob("*_dataset.pkl")))
print(f"Total de PKL encontrados: {len(pkl_files)}")

# Accumulated results
resultados = []
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
csv_path_comma = Path(RESULTS_DIR) / "resultados_autosklearn_comma1.csv"
csv_path_excel = Path(RESULTS_DIR) / "resultados_autosklearn_excel1.csv"

failed = []

# Main loop
for pkl in pkl_files:
    dataset_name = pkl.stem.replace("_dataset", "")
    print(f"\n=== Processando dataset: {dataset_name} ===")

    # Wall clock timer
    start_wall_clock = time.time()

    entry = {
        "dataset": dataset_name,
        "n_train": None,
        "n_test": None,
        "n_features": None,
        "n_classes": None,
        "accuracy": np.nan,
        "auc_ovo": np.nan,
        "gmean": np.nan,
        "cross_entropy": np.nan,
        # Timing columns
        "training_time": 0,  # Time for .fit()
        "prediction_time": 0,  # Time for .predict() and .predict_proba()
        "total_time": 0,  # Sum of both above
        "time_seconds": 0,  # Wall clock time (loading + proc)
        "error": None,
    }

    try:
        # Load pre-processed arrays
        X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
            name=dataset_name
        )

        entry["n_train"] = X_train.shape[0]
        entry["n_test"] = X_test.shape[0]
        entry["n_features"] = X_train.shape[1]
        entry["n_classes"] = len(np.unique(y_train))

        # Configura Auto-sklearn
        model = AutoSklearnClassifier(
            time_limit=1200,
            n_jobs=-1,
            random_state=SEED,
        )

        # --- 1. TREINAMENTO (FIT) ---
        print("Iniciando treinamento...")
        start_fit = time.time()  ### <--- Inicia cronômetro de treino
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit
        print(f"Treino concluído em {fit_time:.2f}s")

        # --- 2. PREVISÃO (PREDICTION) ---
        print("Fazendo previsões...")
        start_pred = time.time()  ### <--- Inicia cronômetro de predição

        # Predições de classe
        y_pred = model.predict(X_test)

        # Predições de probabilidade (com tratamento de erro)
        try:
            y_pred_proba = model.predict_proba(X_test)
            # garantir shape correto
            if y_pred_proba.ndim == 1:
                y_pred_proba = y_pred_proba.reshape(-1, 1)
        except Exception as e:
            # se não conseguiu predict_proba, cria matriz one-hot aproximada
            print("warning: predict_proba falhou:", e)
            n_samples = len(y_pred)
            n_classes = entry["n_classes"] or len(
                np.unique(np.concatenate([y_train, y_pred]))
            )
            y_pred_proba = np.zeros((n_samples, n_classes))
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            for i, lab in enumerate(y_pred):
                idx = label_to_idx.get(lab)
                if idx is not None and idx < n_classes:
                    y_pred_proba[i, idx] = 1.0

        pred_time = time.time() - start_pred  ### <--- Finaliza cronômetro de predição
        print(f"Predição concluída em {pred_time:.4f}s")

        # --- 3. TEMPO TOTAL ---
        total_time = fit_time + pred_time  ### <--- Soma os tempos

        # Atualiza o dicionário com os tempos
        entry["training_time"] = fit_time
        entry["prediction_time"] = pred_time
        entry["total_time"] = total_time

        # Calcula métricas de qualidade
        metrics = calculate_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            n_classes=entry["n_classes"],
        )
        entry.update(metrics)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Erro no dataset {dataset_name}: {e}")
        print(tb)
        entry["error"] = str(e)[:1000]
        failed.append(dataset_name)

    finally:
        # Tempo "Wall Clock" (inclui carregamento de dados)
        entry["time_seconds"] = round(time.time() - start_wall_clock, 2)
        resultados.append(entry)

        # Salva incremental
        df_now = pd.DataFrame(resultados)

        # Reordenar colunas para ficar bonito no CSV (opcional)
        cols_order = [
            "dataset",
            "accuracy",
            "auc_ovo",
            "gmean",
            "cross_entropy",
            "training_time",
            "prediction_time",
            "total_time",
            "time_seconds",
            "n_train",
            "n_test",
            "n_features",
            "n_classes",
            "error",
        ]
        # Filtra apenas colunas que existem no df_now
        cols_to_save = [c for c in cols_order if c in df_now.columns]
        # Adiciona quaisquer outras colunas extras no final
        remaining = [c for c in df_now.columns if c not in cols_to_save]
        df_now = df_now[cols_to_save + remaining]

        df_now.to_csv(csv_path_comma, index=False)
        df_now.to_csv(csv_path_excel, index=False, sep=";")

        print(
            f"Salvou resultado parcial para {dataset_name}. Total time (modelo): {entry.get('total_time', 0):.2f}s"
        )

# ---------- resumo ----------
print("\n=== EXECUÇÃO FINALIZADA ===")
print(f"Total PKL processados: {len(pkl_files)}")
print(f"Total datasets com erro: {len(failed)}")
if failed:
    print("Datasets com erro:", failed)
