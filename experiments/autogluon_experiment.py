import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy"""
    return float(accuracy_score(y_true, y_pred))


def calculate_auc_ovo(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_classes: int
) -> float:
    """
    Calculate AUC using One-vs-One strategy

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes

    Returns:
        AUC score (float)
    """
    try:
        if n_classes > 2:
            auc = roc_auc_score(
                y_true, y_pred_proba, multi_class="ovo", average="macro"
            )
        else:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        return float(auc)
    except Exception:
        return np.nan


def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate G-Mean (geometric mean of sensitivities/recalls)

    This is the geometric mean of recall for each class.
    Useful for imbalanced datasets.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        G-Mean score (float)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate sensitivity (recall) for each class
    sensitivities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)

    # Geometric mean
    gmean = np.power(np.prod(sensitivities), 1.0 / len(sensitivities))
    return float(gmean)


def calculate_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate cross-entropy loss (log loss)

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        Cross-entropy loss (float)
    """
    return float(log_loss(y_true, y_pred_proba))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "auc_ovo": calculate_auc_ovo(y_true, y_pred_proba, n_classes),
        "gmean": calculate_gmean(y_true, y_pred),
        "cross_entropy": calculate_cross_entropy(y_true, y_pred_proba),
    }

    return metrics

from autogluon.tabular import TabularPredictor
import time
import traceback

# Criar diretório para salvar resultados
results_dir = Path("AutogluonResults")
results_dir.mkdir(exist_ok=True)

# Lista para armazenar resultados de todos os datasets
all_results = []

# Quando pkl_files' e 'processor' já estejam definidos anteriormente
for pkl_file in pkl_files:
    dataset_name = pkl_file.stem.replace("_dataset", "")
    print(f"\n{'='*60}")
    print(f"Treinando AutoGluon no dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Carregar dados processados
        X_train, y_train, X_test, y_test = processor.load_or_process_dataset(dataset_name)

        # Converter para DataFrame (AutoGluon precisa de DataFrame)
        df_train = pd.DataFrame(X_train)
        df_train['label'] = y_train

        df_test = pd.DataFrame(X_test)
        df_test['label'] = y_test

        # Configurar o modelo
        model_path = f"AutogluonModels/{dataset_name}/"
        predictor = TabularPredictor(label="label", path=model_path)

        # --- 1. TREINAMENTO (FIT) ---
        print("Iniciando treinamento...")
        start_fit = time.time()

        predictor.fit(
            df_train,
            presets="medium_quality"
        )

        fit_time = time.time() - start_fit
        print(f"Treinamento concluído em {fit_time:.2f} segundos")

        # --- 2. PREVISÃO (PREDICTION) ---
        print("Fazendo previsões...")
        start_pred = time.time()

        y_pred = predictor.predict(df_test)
        y_pred_proba = predictor.predict_proba(df_test)

        prediction_time = time.time() - start_pred
        print(f"Previsão concluída em {prediction_time:.4f} segundos")

        # --- 3. TEMPO TOTAL ---
        total_time = fit_time + prediction_time

        # Converter para numpy arrays
        y_true_np = df_test['label'].values
        y_pred_np = y_pred.values
        y_pred_proba_np = y_pred_proba.values

        # Calcular número de classes
        n_classes = len(np.unique(y_train))

        # Calcular todas as métricas
        print("Calculando métricas...")
        metrics = calculate_all_metrics(
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_pred_proba=y_pred_proba_np,
            n_classes=n_classes
        )

        # Adicionar informações do dataset aos resultados
        dataset_result = {
            'dataset': dataset_name,
            # Métricas de tempo
            'training_time': fit_time,
            'prediction_time': prediction_time,
            'total_time': total_time,
            # Metadados
            'n_classes': n_classes,
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'n_features': X_train.shape[1]
        }
        dataset_result.update(metrics)

        # Exibir resultados
        print(f"\nResultados para {dataset_name}:")
        print(f"  Acurácia: {metrics['accuracy']:.4f}")
        print(f"  AUC OVO: {metrics['auc_ovo']:.4f}")
        print(f"  G-Mean: {metrics['gmean']:.4f}")
        print(f"  Cross-Entropy: {metrics['cross_entropy']:.4f}")
        print(f"  Tempo Treino: {fit_time:.2f}s | Predição: {prediction_time:.4f}s | Total: {total_time:.2f}s")

        # Adicionar à lista de resultados
        all_results.append(dataset_result)

        # Salvar resultados individuais
        result_df = pd.DataFrame([dataset_result])
        result_file = results_dir / f"{dataset_name}_results.csv"
        result_df.to_csv(result_file, index=False)
        print(f"Resultados salvos em: {result_file}")

        # Mostrar leaderboard do AutoGluon
        try:
            leaderboard = predictor.leaderboard(df_test, silent=True)
            print(f"\nTop modelos no leaderboard para {dataset_name}:")
            print(leaderboard.head(5))
        except Exception as e:
            print(f"Erro ao gerar leaderboard: {e}")

    except Exception as e:
        print(f"ERRO em {dataset_name}: {e}")
        print(traceback.format_exc())

        # Registrar resultado com erro
        error_result = {
            'dataset': dataset_name,
            'training_time': np.nan,
            'prediction_time': np.nan,
            'total_time': np.nan,
            'accuracy': np.nan,
            'error': str(e)
        }
        all_results.append(error_result)

# Salvar resultados consolidados
if all_results:
    print(f"\n{'='*80}")
    print("RESUMO DE TODOS OS DATASETS")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)
    summary_file = results_dir / "all_datasets_results.csv"
    results_df.to_csv(summary_file, index=False)
    print(f"Resultados consolidados salvos em: {summary_file}")

    # Mostrar tabela resumo
    print("\nTabela de resultados:")
    # Colunas para exibir no print final
    display_cols = ['dataset', 'accuracy', 'auc_ovo', 'gmean', 'cross_entropy', 'training_time', 'prediction_time', 'total_time']
    available_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df[available_cols].round(4))

else:
    print("Nenhum resultado foi gerado.")

