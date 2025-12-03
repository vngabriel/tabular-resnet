import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from data.data import DataProcessor
from evaluation.metrics import calculate_all_metrics

# Configuration
SEED = 123
CACHE_DIR = "./data/openml_cache"
RESULTS_DIR = "./results/autogluon__"

processor = DataProcessor(seed=SEED, cache_dir=CACHE_DIR)

# Find all processed PKL files
pkl_files = list(Path(CACHE_DIR).glob("*_dataset.pkl"))
print(f"Total de PKL encontrados: {len(pkl_files)}")

# Create directory for results
results_dir = Path(RESULTS_DIR)
results_dir.mkdir(parents=True, exist_ok=True)

# List to store results from all datasets
all_results = []

# Main loop
for pkl_file in pkl_files:
    dataset_name = pkl_file.stem.replace("_dataset", "")
    print(f"\n{'=' * 60}")
    print(f"Treinando AutoGluon no dataset: {dataset_name}")
    print(f"{'=' * 60}")

    try:
        # Load preprocessed data
        X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
            dataset_name
        )

        # Convert to DataFrame (AutoGluon needs DataFrames)
        df_train = pd.DataFrame(X_train)
        df_train["label"] = y_train

        df_test = pd.DataFrame(X_test)
        df_test["label"] = y_test

        # Configure model
        model_path = f"AutogluonModels/{dataset_name}/"
        predictor = TabularPredictor(label="label", path=model_path)

        # --- 1. TRAINING (FIT) ---
        print("Iniciando treinamento...")
        start_fit = time.time()

        predictor.fit(df_train, presets="medium_quality")

        fit_time = time.time() - start_fit
        print(f"Treinamento concluído em {fit_time:.2f} segundos")

        # --- 2. PREDICTION ---
        print("Fazendo previsões...")
        start_pred = time.time()

        y_pred = predictor.predict(df_test)
        y_pred_proba = predictor.predict_proba(df_test)

        prediction_time = time.time() - start_pred
        print(f"Previsão concluída em {prediction_time:.4f} segundos")

        # --- 3. TOTAL TIME ---
        total_time = fit_time + prediction_time

        # Convert to numpy arrays
        y_true_np = df_test["label"].values
        y_pred_np = y_pred.values
        y_pred_proba_np = y_pred_proba.values

        # Calculate number of classes
        n_classes = len(np.unique(y_train))

        # Calculate all metrics
        print("Calculando métricas...")
        metrics = calculate_all_metrics(
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_pred_proba=y_pred_proba_np,
            n_classes=n_classes,
        )

        # Add dataset info to results
        dataset_result = {
            "dataset": dataset_name,
            # Timing metrics
            "training_time": fit_time,
            "prediction_time": prediction_time,
            "total_time": total_time,
            # Metadata
            "n_classes": n_classes,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "n_features": X_train.shape[1],
        }
        dataset_result.update(metrics)

        # Display results
        print(f"\nResultados para {dataset_name}:")
        print(f"  Acurácia: {metrics['accuracy']:.4f}")
        print(f"  AUC OVO: {metrics['auc_ovo']:.4f}")
        print(f"  G-Mean: {metrics['gmean']:.4f}")
        print(f"  Cross-Entropy: {metrics['cross_entropy']:.4f}")
        print(
            f"  Tempo Treino: {fit_time:.2f}s | Predição: {prediction_time:.4f}s | Total: {total_time:.2f}s"
        )

        # Add to results list
        all_results.append(dataset_result)

        # Save individual results
        result_df = pd.DataFrame([dataset_result])
        result_file = results_dir / f"{dataset_name}_results.csv"
        result_df.to_csv(result_file, index=False)
        print(f"Resultados salvos em: {result_file}")

        # Show AutoGluon leaderboard
        try:
            leaderboard = predictor.leaderboard(df_test, silent=True)
            print(f"\nTop modelos no leaderboard para {dataset_name}:")
            print(leaderboard.head(5))
        except Exception as e:
            print(f"Erro ao gerar leaderboard: {e}")

    except Exception as e:
        print(f"ERRO em {dataset_name}: {e}")
        print(traceback.format_exc())

        # Register error result
        error_result = {
            "dataset": dataset_name,
            "training_time": np.nan,
            "prediction_time": np.nan,
            "total_time": np.nan,
            "accuracy": np.nan,
            "error": str(e),
        }
        all_results.append(error_result)

# Save consolidated results
if all_results:
    print(f"\n{'=' * 80}")
    print("RESUMO DE TODOS OS DATASETS")
    print(f"{'=' * 80}")

    results_df = pd.DataFrame(all_results)
    summary_file = results_dir / "all_datasets_results.csv"
    results_df.to_csv(summary_file, index=False)
    print(f"Resultados consolidados salvos em: {summary_file}")

    # Show summary table
    print("\nTabela de resultados:")
    # Columns to display in final print
    display_cols = [
        "dataset",
        "accuracy",
        "auc_ovo",
        "gmean",
        "cross_entropy",
        "training_time",
        "prediction_time",
        "total_time",
    ]
    available_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df[available_cols].round(4))

else:
    print("Nenhum resultado foi gerado.")
