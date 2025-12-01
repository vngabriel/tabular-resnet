import os
import glob
import traceback
import pandas as pd
from autogluon.tabular import TabularPredictor
import time
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

base_path = "/content/datasets"

csv_files = sorted(glob.glob(os.path.join(base_path, "*.csv")))
datasets_list = []

for path in csv_files:
  name = os.path.splitext(os.path.basename(path))[0]

  entry = {
      "name": name,
      "path": path,
      "label": "target"
  }
  datasets_list.append(entry)

print("total de datasets encontrados:", len(datasets_list))
for d in datasets_list:
  print(d["name"])

class DataProcessor:
    """Handles data preprocessing, missing value handling, and train/test splitting"""

    def __init__(self, seed: int, cache_dir: str = "./data/openml_cache"):
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def handle_missing_values(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "impute",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Handle missing values in features and target

        Args:
            X: Feature dataframe
            y: Target series
            strategy: 'impute' or 'drop'

        Returns:
            Cleaned X and y
        """
        # Drop columns that are entirely null
        null_cols = X.columns[X.isnull().all()].tolist()
        if null_cols:
            X = X.drop(columns=null_cols)
            print(f"Dropped {len(null_cols)} columns with all null values: {null_cols}")

        if strategy == "drop":
            # Drop rows with any null values
            if X.isnull().any().any() or y.isnull().any():
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                rows_dropped = (~mask).sum()
                X = X[mask].reset_index(drop=True)
                y = y[mask].reset_index(drop=True)
                print(f"Dropped {rows_dropped} rows with null values")

        elif strategy == "impute":
            # Impute features
            if X.isnull().any().any():
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                if numeric_cols:
                    num_null_count = X[numeric_cols].isnull().sum().sum()
                    if num_null_count > 0:
                        num_imputer = SimpleImputer(strategy="median")
                        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
                        print(f"Imputed {num_null_count} numerical nulls (median)")

                if categorical_cols:
                    cat_null_count = X[categorical_cols].isnull().sum().sum()
                    if cat_null_count > 0:
                        cat_imputer = SimpleImputer(strategy="most_frequent")
                        X[categorical_cols] = cat_imputer.fit_transform(
                            X[categorical_cols]
                        )
                        print(
                            f"Imputed {cat_null_count} categorical nulls (most_frequent)"
                        )

            # Always drop rows with null target
            if y.isnull().any():
                mask = ~y.isnull()
                rows_dropped = (~mask).sum()
                X = X[mask].reset_index(drop=True)
                y = y[mask].reset_index(drop=True)
                print(f"Dropped {rows_dropped} rows with null target")

        return X, y

    def preprocess_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        scale_categorical: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Preprocess features: encode categoricals, scale, and encode target

        This applies the same preprocessing as in experiment.py:
        - Encode target labels
        - One-hot encode categorical features
        - Scale features with StandardScaler

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            scale_categorical: If True, scale one-hot encoded categorical features.
                             If False, only scale numerical features (keeps categorical as 0/1).
                             Default False

        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, n_classes)
            All as numpy arrays ready for model training
        """
        # Encode target labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        n_classes = len(label_encoder.classes_)
        print(f"Number of classes: {n_classes}")

        # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Numerical features: {len(numerical_cols)}")

        # Handle categorical features
        if categorical_cols:
            categorical_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            X_train_cat = categorical_encoder.fit_transform(X_train[categorical_cols])
            X_test_cat = categorical_encoder.transform(X_test[categorical_cols])

            # Get numerical features
            X_train_num = (
                X_train[numerical_cols].values
                if numerical_cols
                else np.array([]).reshape(len(X_train), 0)
            )
            X_test_num = (
                X_test[numerical_cols].values
                if numerical_cols
                else np.array([]).reshape(len(X_test), 0)
            )

            # Scale based on flag
            if scale_categorical:
                # Scale everything together (numerical + categorical)
                X_train_combined = np.hstack([X_train_num, X_train_cat])
                X_test_combined = np.hstack([X_test_num, X_test_cat])  # type: ignore

                scaler = StandardScaler()
                X_train_processed = scaler.fit_transform(X_train_combined)
                X_test_processed = scaler.transform(X_test_combined)

                print("    Scaled numerical + categorical features together")
            else:
                # Scale only numerical features, keep categorical as 0/1
                if numerical_cols:
                    scaler = StandardScaler()
                    X_train_num_scaled = scaler.fit_transform(X_train_num)
                    X_test_num_scaled = scaler.transform(X_test_num)
                else:
                    X_train_num_scaled = X_train_num
                    X_test_num_scaled = X_test_num

                X_train_processed = np.hstack([X_train_num_scaled, X_train_cat])
                X_test_processed = np.hstack([X_test_num_scaled, X_test_cat])  # type: ignore

                print("    Scaled only numerical features (categorical kept as 0/1)")
        else:
            # No categorical features, just scale numerical
            X_train_processed = X_train[numerical_cols].values
            X_test_processed = X_test[numerical_cols].values

            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_processed)
            X_test_processed = scaler.transform(X_test_processed)

            print("    Scaled numerical features only")

        print(f"Total features after preprocessing: {X_train_processed.shape[1]}")

        return (
            X_train_processed,
            X_test_processed,
            y_train_encoded,
            y_test_encoded,
            n_classes,
        )  # type: ignore

    def process_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        test_size: float = 0.3,
        missing_value_strategy: str = "impute",
        scale_categorical: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single dataset: split train/test, handle missing values and preprocess features

        IMPORTANT: Data split must be done BEFORE handling missing values and preprocessing to avoid data leakage!
        - Split first (to separate train/test)
        - Then impute using ONLY training data statistics
        - Apply those statistics to test data

        Args:
            df: Raw dataframe (target as last column)
            name: Dataset name
            test_size: Test split size
            missing_value_strategy: 'impute' or 'drop'
            scale_categorical: Whether to scale one-hot encoded categorical features

        Returns:
            X_train, y_train, X_test, y_test
        """
        processed_pkl_path = self.cache_dir / f"{name}_dataset.pkl"

        # Check if already processed
        if processed_pkl_path.exists():
            print(f"Loading processed dataset from cache: {name}")
            with open(processed_pkl_path, "rb") as f:
                return pickle.load(f)

        print(f"Processing dataset: {name}")

        # Separate features and target
        X = df.drop(columns=["target"])
        y = df["target"]

        print(f"Original shape: {X.shape}")
        print(f"Classes: {y.nunique()}")

        # STEP 1: Drop columns that are entirely null (do this before split)
        null_cols = X.columns[X.isnull().all()].tolist()
        if null_cols:
            X = X.drop(columns=null_cols)
            print(f"Dropped {len(null_cols)} columns with all null values: {null_cols}")

        # STEP 2: Drop rows where TARGET is null (do this before split)
        if y.isnull().any():
            mask = ~y.isnull()
            rows_dropped = (~mask).sum()
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
            print(f"Dropped {rows_dropped} rows with null target")

        # STEP 3: Split train/test BEFORE handling missing values in features
        # This prevents data leakage!
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.seed, stratify=y
            )
        except ValueError:
            # If stratification fails (e.g., too few samples in a class), don't stratify
            print("Warning: Stratification failed, splitting without stratify")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.seed
            )

        print(f"Split: Train={X_train.shape}, Test={X_test.shape}")

        # STEP 4: Handle missing values in FEATURES (after split, to avoid leakage)
        if missing_value_strategy == "drop":
            # Drop rows with any null values
            if X_train.isnull().any().any():
                mask = ~X_train.isnull().any(axis=1)
                rows_dropped = (~mask).sum()
                X_train = X_train[mask].reset_index(drop=True)
                y_train = y_train[mask].reset_index(drop=True)
                print(f"Dropped {rows_dropped} training rows with null values")

            if X_test.isnull().any().any():
                mask = ~X_test.isnull().any(axis=1)
                rows_dropped = (~mask).sum()
                X_test = X_test[mask].reset_index(drop=True)
                y_test = y_test[mask].reset_index(drop=True)
                print(f"Dropped {rows_dropped} test rows with null values")

        elif missing_value_strategy == "impute":
            # Impute using ONLY training data statistics (no data leakage!)
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if numeric_cols and X_train[numeric_cols].isnull().any().any():
                num_null_count = X_train[numeric_cols].isnull().sum().sum()
                if num_null_count > 0:
                    num_imputer = SimpleImputer(strategy="median")
                    X_train[numeric_cols] = num_imputer.fit_transform(
                        X_train[numeric_cols]
                    )
                    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
                    print(
                        f"Imputed {num_null_count} numerical nulls (median from train)"
                    )

            if categorical_cols and X_train[categorical_cols].isnull().any().any():
                cat_null_count = X_train[categorical_cols].isnull().sum().sum()
                if cat_null_count > 0:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    X_train[categorical_cols] = cat_imputer.fit_transform(
                        X_train[categorical_cols]
                    )
                    X_test[categorical_cols] = cat_imputer.transform(
                        X_test[categorical_cols]
                    )
                    print(
                        f"Imputed {cat_null_count} categorical nulls (most_frequent from train)"
                    )

        print(f"After cleaning: Train={X_train.shape}, Test={X_test.shape}\n")

        (
            X_train_processed,
            X_test_processed,
            y_train_encoded,
            y_test_encoded,
            _,
        ) = self.preprocess_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scale_categorical=scale_categorical,
        )

        result = (X_train_processed, y_train_encoded, X_test_processed, y_test_encoded)

        # Cache processed dataset
        with open(processed_pkl_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved processed dataset: {processed_pkl_path}")

        return result

    def load_or_process_dataset(
        self,
        name: str,
        raw_df: pd.DataFrame | None = None,
        test_size: float = 0.3,
        missing_value_strategy: str = "impute",
        scale_categorical: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed dataset from cache or process raw dataframe

        Args:
            name: Dataset name
            raw_df: Raw dataframe (if None, will try to load from CSV)
            test_size: Test split size
            missing_value_strategy: 'impute' or 'drop'
            scale_categorical: Whether to scale one-hot encoded categorical features

        Returns:
            X_train, y_train, X_test, y_test
        """
        processed_pkl_path = self.cache_dir / f"{name}_dataset.pkl"

        # Try to load processed dataset
        if processed_pkl_path.exists():
            print(f"Loading processed dataset from cache: {name}")
            with open(processed_pkl_path, "rb") as f:
                return pickle.load(f)

        # Try to load raw CSV if raw_df not provided
        if raw_df is None:
            raw_csv_path = self.cache_dir / f"{name}_dataset.csv"
            if not raw_csv_path.exists():
                raise ValueError(
                    f"No raw data found for {name}. Please download first."
                )
            print(f"Loading raw CSV: {name}")
            raw_df = pd.read_csv(raw_csv_path)

        # Process the dataset
        return self.process_dataset(
            raw_df,
            name,
            test_size=test_size,
            missing_value_strategy=missing_value_strategy,
            scale_categorical=scale_categorical,
        )

print("\n" + "=" * 80)
    print("PATTERN 2: Process all raw CSVs when OpenML is down")
    print("=" * 80)

    # When OpenML is down but you have raw CSV files saved
    # This will find all raw CSVs and process them to PKL

    SEED = 42
    processor = DataProcessor(seed=SEED, cache_dir=base_path)

    # Find all raw CSV files in the cache directory
    raw_csv_files = list(processor.cache_dir.glob("*_dataset.csv"))

    if raw_csv_files:
        print(f"Found {len(raw_csv_files)} raw CSV files to process\n")

        for csv_file in raw_csv_files:
            # Extract dataset name from filename
            # Format: {dataset_name}_dataset.csv
            dataset_name = csv_file.stem.replace("_dataset", "")

            try:
                print(f"\nProcessing: {dataset_name}")

                # Load raw CSV
                raw_df = pd.read_csv(csv_file)

                # Process dataset (split, impute, encode, scale)
                X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
                    name=dataset_name,
                    test_size=0.3,
                    missing_value_strategy="impute",
                    scale_categorical=False,  # Set to True to scale categorical features
                )

                print(f"  ✓ Processed and saved PKL for {dataset_name}")
                print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
                print(f"    Classes: {len(set(y_train))}")

            except Exception as e:
                print(f"  ✗ Error processing {dataset_name}: {e}")

    else:
        print("No raw CSV files found in cache directory")
        print(f"Looking in: {processor.cache_dir}")

    # ==============================================================================
    # Pattern 3: Fast loading from cached PKL files
    # (You have already processed datasets, just load them for training)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("PATTERN 3: Fast loading from cached PKL files")
    print("=" * 80)

    # When you have already processed datasets and just want to load them quickly
    # This is the fastest way - directly loads preprocessed numpy arrays

    processor = DataProcessor(seed=SEED, cache_dir=base_path)

    # Find all processed PKL files
    pkl_files = list(processor.cache_dir.glob("*_dataset.pkl"))

    if pkl_files:
        print(f"Found {len(pkl_files)} processed PKL files\n")

        for pkl_file in pkl_files:
            # Extract dataset name from filename
            # Format: {dataset_name}_dataset.pkl
            dataset_name = pkl_file.stem.replace("_dataset", "")

            try:
                print(f"Loading: {dataset_name}")

                # Load preprocessed dataset (fastest - directly from PKL)
                X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
                    name=dataset_name
                )

                print("  ✓ Loaded from cache")
                print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
                print(f"    Classes: {len(set(y_train))}")
                print("    Ready for training!\n")

                # Now you can directly train your model:
                # model.fit(X_train, y_train)
                # predictions = model.predict(X_test)

            except Exception as e:
                print(f"  ✗ Error loading {dataset_name}: {e}\n")

    else:
        print("No processed PKL files found in cache directory")
        print(f"Looking in: {processor.cache_dir}")
        print("\nRun Pattern 1 or 2 first to process datasets")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import calculate_all_metrics



from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback

# Criar diretório para salvar resultados
results_dir = Path("AutogluonResults")
results_dir.mkdir(exist_ok=True)

# Lista para armazenar resultados de todos os datasets
all_results = []

# Supondo que 'pkl_files' e 'processor' já estejam definidos anteriormente no seu script
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

