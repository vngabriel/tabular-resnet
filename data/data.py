import os
import pickle
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

os.environ["OPENML_SKIP_PARQUET"] = "true"


class OpenMLDownloader:
    """Handles downloading and caching raw datasets from OpenML"""

    def __init__(self, cache_dir: str = "./data/openml_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.benchmark_id = 99  # OpenML-CC18
        self.datasets_info = None

    def get_smallest_datasets(self, n_datasets: int = 30) -> pd.DataFrame:
        """Get the N smallest datasets from OpenML-CC18 benchmark by number of samples"""

        print("Fetching OpenML-CC18 benchmark tasks...")
        benchmark = openml.study.get_suite(self.benchmark_id)

        if not benchmark.tasks:
            raise ValueError("No tasks found in the OpenML-CC18 benchmark")

        print(f"Collecting dataset information for {len(benchmark.tasks)} tasks...")
        datasets_info = []

        for idx, task_id in enumerate(benchmark.tasks):
            try:
                task = openml.tasks.get_task(task_id)
                dataset = openml.datasets.get_dataset(
                    task.dataset_id, download_data=True
                )

                if dataset is None or dataset.qualities is None:
                    print(f"Warning: Dataset {task.dataset_id} has no qualities info")
                    continue

                datasets_info.append(
                    {
                        "task_id": task_id,
                        "dataset_id": task.dataset_id,
                        "name": dataset.name,
                        "target": dataset.default_target_attribute,
                        "n_samples": int(dataset.qualities.get("NumberOfInstances", 0)),
                        "n_features": int(dataset.qualities.get("NumberOfFeatures", 0)),
                        "n_classes": int(dataset.qualities.get("NumberOfClasses", 0)),
                    }
                )

                if (idx + 1) % 5 == 0:
                    print(f"Processed {idx + 1}/{len(benchmark.tasks)} tasks...")

            except Exception as e:
                print(f"Warning: Could not process task {task_id}: {e}")
                continue

        # Create dataframe and filter valid datasets
        df = pd.DataFrame(datasets_info)
        df = df[(df["n_samples"] > 0) & (df["n_features"] > 0) & (df["n_classes"] > 1)]

        # Select smallest by number of samples
        smallest = df.nsmallest(n_datasets, "n_samples").reset_index(drop=True)
        self.datasets_info = smallest

        # Save info
        info_path = self.cache_dir / "datasets_info.csv"
        smallest.to_csv(info_path, index=False)

        print(f"\n{'=' * 80}")
        print(f"Selected {len(smallest)} smallest datasets (by n_samples):")
        print(f"{'=' * 80}")
        print(smallest[["name", "n_samples", "n_features", "n_classes"]])
        print(f"{'=' * 80}\n")

        return smallest

    def download_dataset(
        self, dataset_id: int, name: str, target: str | None = None
    ) -> pd.DataFrame:
        """
        Download a single dataset from OpenML and save as CSV

        Args:
            dataset_id: OpenML dataset ID
            name: Dataset name
            target: Target column name (if None, will use default_target_attribute)

        Returns:
            DataFrame with features and target (target is last column)
        """
        raw_csv_path = self.cache_dir / f"{name}_dataset.csv"

        # Check if already downloaded
        if raw_csv_path.exists():
            print(f"Loading cached raw CSV: {name}")
            return pd.read_csv(raw_csv_path)

        print(f"Downloading from OpenML: {name} (ID: {dataset_id})")

        # Download dataset
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=target or dataset.default_target_attribute,
        )

        # Combine X and y into single dataframe (target as last column)
        df = X.copy()
        df["target"] = y

        # Save raw CSV
        df.to_csv(raw_csv_path, index=False)  # type: ignore
        print(f"Saved raw CSV: {raw_csv_path}")

        return df  # type: ignore

    def load_or_download_dataset(
        self, dataset_id: int, name: str, target: str | None = None
    ) -> pd.DataFrame:
        """
        Load dataset from cache or download from OpenML

        Args:
            dataset_id: OpenML dataset ID
            name: Dataset name
            target: Target column name

        Returns:
            Raw dataframe with target as last column
        """
        raw_csv_path = self.cache_dir / f"{name}_dataset.csv"

        if raw_csv_path.exists():
            print(f"Loading cached raw CSV: {name}")
            return pd.read_csv(raw_csv_path)
        else:
            return self.download_dataset(dataset_id, name, target)

    def download_all_datasets(self) -> dict[str, pd.DataFrame]:
        """Download all datasets from the benchmark"""
        datasets_info_path = self.cache_dir / "datasets_info.csv"
        if datasets_info_path.exists():
            self.datasets_info = pd.read_csv(datasets_info_path)

        if self.datasets_info is None:
            raise ValueError("Run get_smallest_datasets() first")

        datasets = {}
        print(f"\nDownloading {len(self.datasets_info)} datasets...\n")

        for idx, row in self.datasets_info.iterrows():
            try:
                df = self.load_or_download_dataset(
                    dataset_id=row["dataset_id"],
                    name=row["name"],
                    target=row.get("target"),
                )
                datasets[row["name"]] = df
                print(
                    f"[{idx + 1}/{len(self.datasets_info)}] {row['name']}: {df.shape}\n"  # type: ignore
                )

            except Exception as e:
                print(f"Error downloading {row['name']}: {str(e)}\n")

        print(
            f"Successfully downloaded {len(datasets)}/{len(self.datasets_info)} datasets"
        )
        return datasets


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


if __name__ == "__main__":
    SEED = 123

    # ==============================================================================
    # Pattern 1: Complete workflow - Download and Process (OpenML online)
    # ==============================================================================
    print("=" * 80)
    print("PATTERN 1: Complete workflow - Download and Process")
    print("=" * 80)

    # Step 1: Download raw datasets from OpenML (if needed)
    downloader = OpenMLDownloader(cache_dir="./data/openml_cache")

    # Get the smallest datasets from the benchmark
    datasets_info = downloader.get_smallest_datasets()

    # Download all datasets (saves as CSV)
    raw_datasets = downloader.download_all_datasets()

    # Step 2: Process datasets (returns preprocessed numpy arrays ready for training)
    processor = DataProcessor(seed=SEED, cache_dir="./data/openml_cache")

    # Load datasets info
    datasets_info_path = processor.cache_dir / "datasets_info.csv"
    if datasets_info_path.exists():
        datasets_info_df = pd.read_csv(datasets_info_path)

        for idx, row in datasets_info_df.iterrows():
            try:
                print(f"\n{row['name']}:")

                # Process dataset: split, impute, encode, scale
                # Returns numpy arrays ready for model training
                X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
                    name=row["name"],
                    test_size=0.3,
                    missing_value_strategy="impute",  # or 'drop'
                    scale_categorical=False,  # Set to True to scale categorical features
                )

                print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
                print(f"  Classes: {len(set(y_train))}")
                print("  Ready for model training!")

                # Now you can use these for model training:
                # - X_train, X_test: numpy arrays
                # - y_train, y_test: numpy arrays

            except Exception as e:
                print(f"  Error: {e}")

    else:
        print(
            "No datasets_info.csv found. Run downloader.get_smallest_datasets() first."
        )

    # ==============================================================================
    # Pattern 2: Working offline
    # (OpenML is down, you don't have the datasets_info.csv but you have raw CSVs)
    # ==============================================================================
    # print("\n" + "=" * 80)
    # print("PATTERN 2: Process all raw CSVs when OpenML is down")
    # print("=" * 80)

    # # When OpenML is down but you have raw CSV files saved
    # # This will find all raw CSVs and process them to PKL

    # processor = DataProcessor(seed=SEED, cache_dir="./data/openml_cache")

    # # Find all raw CSV files in the cache directory
    # raw_csv_files = list(processor.cache_dir.glob("*_dataset.csv"))

    # if raw_csv_files:
    #     print(f"Found {len(raw_csv_files)} raw CSV files to process\n")

    #     for csv_file in raw_csv_files:
    #         # Extract dataset name from filename
    #         # Format: {dataset_name}_dataset.csv
    #         dataset_name = csv_file.stem.replace("_dataset", "")

    #         try:
    #             print(f"\nProcessing: {dataset_name}")

    #             # Load raw CSV
    #             raw_df = pd.read_csv(csv_file)

    #             # Process dataset (split, impute, encode, scale)
    #             X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
    #                 name=dataset_name,
    #                 test_size=0.3,
    #                 missing_value_strategy="impute",
    #                 scale_categorical=False,  # Set to True to scale categorical features
    #             )

    #             print(f"  ✓ Processed and saved PKL for {dataset_name}")
    #             print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    #             print(f"    Classes: {len(set(y_train))}")

    #         except Exception as e:
    #             print(f"  ✗ Error processing {dataset_name}: {e}")

    # else:
    #     print("No raw CSV files found in cache directory")
    #     print(f"Looking in: {processor.cache_dir}")

    # ==============================================================================
    # Pattern 3: Fast loading from cached PKL files
    # (You have already processed datasets, just load them for training)
    # ==============================================================================
    # print("\n" + "=" * 80)
    # print("PATTERN 3: Fast loading from cached PKL files")
    # print("=" * 80)

    # # When you have already processed datasets and just want to load them quickly
    # # This is the fastest way - directly loads preprocessed numpy arrays

    # processor = DataProcessor(seed=SEED, cache_dir="./data/openml_cache")

    # # Find all processed PKL files
    # pkl_files = list(processor.cache_dir.glob("*_dataset.pkl"))

    # if pkl_files:
    #     print(f"Found {len(pkl_files)} processed PKL files\n")

    #     for pkl_file in pkl_files:
    #         # Extract dataset name from filename
    #         # Format: {dataset_name}_dataset.pkl
    #         dataset_name = pkl_file.stem.replace("_dataset", "")

    #         try:
    #             print(f"Loading: {dataset_name}")

    #             # Load preprocessed dataset (fastest - directly from PKL)
    #             X_train, y_train, X_test, y_test = processor.load_or_process_dataset(
    #                 name=dataset_name
    #             )

    #             print("  ✓ Loaded from cache")
    #             print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    #             print(f"    Classes: {len(set(y_train))}")
    #             print("    Ready for training!\n")

    #             # Now you can directly train your model:
    #             # model.fit(X_train, y_train)
    #             # predictions = model.predict(X_test)

    #         except Exception as e:
    #             print(f"  ✗ Error loading {dataset_name}: {e}\n")

    # else:
    #     print("No processed PKL files found in cache directory")
    #     print(f"Looking in: {processor.cache_dir}")
    #     print("\nRun Pattern 1 or 2 first to process datasets")
