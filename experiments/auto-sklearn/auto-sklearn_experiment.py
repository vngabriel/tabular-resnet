from auto_sklearn2 import AutoSklearnClassifier
from sklearn.metrics import accuracy_score
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path


from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Funções de métricas CORRIGIDAS
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy"""
    return float(accuracy_score(y_true, y_pred))

def calculate_auc_ovo(y_true: np.ndarray, y_pred_proba: np.ndarray, n_classes: int) -> float:
    """Calculate AUC using One-vs-One strategy"""
    try:
        if n_classes > 2:
            auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo", average="macro")
        else:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        return float(auc)
    except Exception as e:
        print(f"Erro no cálculo do AUC: {e}")
        return np.nan

def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate G-Mean (geometric mean of sensitivities/recalls)"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        sensitivities = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sensitivity)
        gmean = np.power(np.prod(sensitivities), 1.0 / len(sensitivities))
        return float(gmean)
    except Exception as e:
        print(f"Erro no cálculo do G-Mean: {e}")
        return np.nan

def calculate_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate cross-entropy loss (log loss)"""
    try:
        return float(log_loss(y_true, y_pred_proba))
    except Exception as e:
        print(f"Erro no cálculo do Cross-Entropy: {e}")
        return np.nan

def calculate_all_metrics_safe(y_true, y_pred, y_pred_proba, n_classes):
    """Versão mais segura que calcula métricas individualmente"""
    metrics = {}

    try:
        metrics["accuracy"] = calculate_accuracy(y_true, y_pred)
    except Exception as e:
        print(f"Erro cálculo accuracy: {e}")
        metrics["accuracy"] = np.nan

    try:
        metrics["gmean"] = calculate_gmean(y_true, y_pred)
    except Exception as e:
        print(f"Erro cálculo gmean: {e}")
        metrics["gmean"] = np.nan

    try:
        metrics["auc_ovo"] = calculate_auc_ovo(y_true, y_pred_proba, n_classes)
    except Exception as e:
        print(f"Erro cálculo auc: {e}")
        metrics["auc_ovo"] = np.nan

    try:
        metrics["cross_entropy"] = calculate_cross_entropy(y_true, y_pred_proba)
    except Exception as e:
        print(f"Erro cálculo cross_entropy: {e}")
        metrics["cross_entropy"] = np.nan

    return metrics

# Função auxiliar para obter probabilidades de forma segura
def safe_predict_proba(model, X, n_classes):
    """Tenta obter probabilidades de forma segura"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # Verificar se as probabilidades somam aproximadamente 1
            if proba.shape[1] == n_classes:
                return proba
            else:
                print(f"Aviso: predict_proba retornou shape {proba.shape}, esperado (n_samples, {n_classes})")
                return None
        else:
            print("Modelo não possui método predict_proba")
            return None
    except Exception as e:
        print(f"Erro ao obter probabilidades: {e}")
        return None

# Função para criar probabilidades a partir de labels (fallback)
def create_proba_from_predictions(y_pred, n_classes):
    """Cria array de probabilidades one-hot a partir das previsões"""
    proba = np.zeros((len(y_pred), n_classes))
    for i, pred in enumerate(y_pred):
        proba[i, pred] = 1.0
    return proba

import time
import numpy as np
import os
import glob
import traceback
import pandas as pd
import traceback
from pathlib import Path
from auto_sklearn2 import AutoSklearnClassifier
# ---------- configuração ----------
base_path = Path(base_path) # Defina seu base_path aqui
pkl_files = sorted(list(base_path.glob("*_dataset.pkl")))
print(f"Total de PKL encontrados: {len(pkl_files)}")

# resultado acumulado
resultados = []
csv_path_comma = base_path / "resultados_autosklearn_comma1.csv"
csv_path_excel = base_path / "resultados_autosklearn_excel1.csv"

failed = []

# ---------- loop ----------
for pkl in pkl_files:
    dataset_name = pkl.stem.replace("_dataset", "")
    print(f"\n=== Processando dataset: {dataset_name} ===")

    # Cronômetro geral (Wall clock)
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

        # Novas colunas de tempo
        "training_time": 0,    ### <--- Tempo do .fit()
        "prediction_time": 0,  ### <--- Tempo do .predict() e .predict_proba()
        "total_time": 0,       ### <--- Soma dos dois acima
        "time_seconds": 0,     ### <--- Tempo de execução do script (loading + proc)

        "error": None,
    }

    try:
        # Carrega arrays pré-processados
        X_train, y_train, X_test, y_test = processor.load_or_process_dataset(name=dataset_name)

        entry["n_train"] = X_train.shape[0]
        entry["n_test"] = X_test.shape[0]
        entry["n_features"] = X_train.shape[1]
        entry["n_classes"] = len(np.unique(y_train))

        # Configura Auto-sklearn
        model = AutoSklearnClassifier(
            time_limit=1200,
            n_jobs=-1,
        )

        # --- 1. TREINAMENTO (FIT) ---
        print("Iniciando treinamento...")
        start_fit = time.time()          ### <--- Inicia cronômetro de treino
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit
        print(f"Treino concluído em {fit_time:.2f}s")

        # --- 2. PREVISÃO (PREDICTION) ---
        print("Fazendo previsões...")
        start_pred = time.time()         ### <--- Inicia cronômetro de predição

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
            n_classes = entry["n_classes"] or len(np.unique(np.concatenate([y_train, y_pred])))
            y_pred_proba = np.zeros((n_samples, n_classes))
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            for i, lab in enumerate(y_pred):
                idx = label_to_idx.get(lab)
                if idx is not None and idx < n_classes:
                    y_pred_proba[i, idx] = 1.0

        pred_time = time.time() - start_pred ### <--- Finaliza cronômetro de predição
        print(f"Predição concluída em {pred_time:.4f}s")

        # --- 3. TEMPO TOTAL ---
        total_time = fit_time + pred_time    ### <--- Soma os tempos

        # Atualiza o dicionário com os tempos
        entry["training_time"] = fit_time
        entry["prediction_time"] = pred_time
        entry["total_time"] = total_time

        # Calcula métricas de qualidade
        metrics = calculate_all_metrics_safe(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            n_classes=entry["n_classes"]
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
            "dataset", "accuracy", "auc_ovo", "gmean", "cross_entropy",
            "training_time", "prediction_time", "total_time", "time_seconds",
            "n_train", "n_test", "n_features", "n_classes", "error"
        ]
        # Filtra apenas colunas que existem no df_now
        cols_to_save = [c for c in cols_order if c in df_now.columns]
        # Adiciona quaisquer outras colunas extras no final
        remaining = [c for c in df_now.columns if c not in cols_to_save]
        df_now = df_now[cols_to_save + remaining]

        df_now.to_csv(csv_path_comma, index=False)
        df_now.to_csv(csv_path_excel, index=False, sep=";")

        print(f"Salvou resultado parcial para {dataset_name}. Total time (modelo): {entry.get('total_time', 0):.2f}s")

# ---------- resumo ----------
print("\n=== EXECUÇÃO FINALIZADA ===")
print(f"Total PKL processados: {len(pkl_files)}")
print(f"Total datasets com erro: {len(failed)}")
if failed:
    print("Datasets com erro:", failed)

