"""
GPU-Accelerated version of experiment_boosting.py

This version supports GPU acceleration for XGBoost, CatBoost, and LightGBM.
Use this file if you have an NVIDIA GPU available.

Usage:
    # Enable GPU
    experiment = BoostingExperiment(..., use_gpu=True)

    # Or via CLI
    python main_boosting_gpu.py --model catboost --use_gpu
"""

import time

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from evaluation.metrics import calculate_all_metrics

# Import boosting libraries
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BoostingExperimentGPU:
    """GPU-accelerated experiment pipeline for Boosting models"""

    def __init__(
        self,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int,
        model_type: str = "xgboost",
        n_trials: int = 50,
        cv_folds: int = 10,
        seed: int = 123,
        use_gpu: bool = False,  # NEW: GPU support
        gpu_id: int = 0,  # NEW: Which GPU to use
    ):
        """
        Initialize experiment with GPU support

        Args:
            ... (same as BoostingExperiment)
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID (0, 1, 2, etc.)
        """
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes

        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.seed = seed

        # GPU settings
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Validate model type
        if self.model_type not in ["xgboost", "catboost", "lightgbm"]:
            raise ValueError(
                f"model_type must be 'xgboost', 'catboost', or 'lightgbm', got {model_type}"
            )

        # Check if the required library is available
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        elif self.model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed")
        elif self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        # Dataset info
        self.n_features = X_train.shape[1]
        self.n_samples_train = len(X_train)
        self.n_samples_test = len(X_test)

        print(f"Dataset: {dataset_name}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {self.n_features}, Classes: {self.n_classes}")
        print(f"  Model: {self.model_type.upper()}")
        print(f"  Device: {'GPU' if self.use_gpu else 'CPU'}")
        if self.use_gpu:
            print(f"  GPU ID: {self.gpu_id}")

        # Results storage
        self.best_params = None
        self.best_model = None

        # Timing
        self.tuning_time = 0
        self.training_time = 0
        self.prediction_time = 0

    def create_model(self, params: dict):
        """Create boosting model with GPU support"""

        if self.model_type == "xgboost":
            # XGBoost GPU settings
            if self.use_gpu:
                params["device"] = "cuda"
                params["tree_method"] = "hist"  # GPU-compatible method

            model = xgb.XGBClassifier(
                **params,
                random_state=self.seed,
                n_jobs=-1 if not self.use_gpu else 1,
                verbosity=0,
            )

        elif self.model_type == "catboost":
            # Ensure bootstrap_type is set for subsample compatibility
            if "subsample" in params and "bootstrap_type" not in params:
                params["bootstrap_type"] = "Bernoulli"

            # CatBoost GPU settings
            model = cb.CatBoostClassifier(
                **params,
                random_state=self.seed,
                task_type="GPU" if self.use_gpu else "CPU",
                devices=str(self.gpu_id) if self.use_gpu else None,
                thread_count=-1 if not self.use_gpu else 1,
                verbose=False,
            )

        elif self.model_type == "lightgbm":
            # LightGBM GPU settings
            if self.use_gpu:
                params["device"] = "gpu"
                params["gpu_platform_id"] = 0
                params["gpu_device_id"] = self.gpu_id

            model = lgb.LGBMClassifier(
                **params,
                random_state=self.seed,
                n_jobs=-1 if not self.use_gpu else 1,
                verbose=-1,
            )

        return model

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict:
        """Define hyperparameter search space for each model type"""

        if self.model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

        elif self.model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                # "bootstrap_type": "Bernoulli",
                # "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            }

        elif self.model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            }

        return params

    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train model with given data"""

        if self.model_type == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        elif self.model_type == "catboost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )

        elif self.model_type == "lightgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

        return model

    def objective(self, trial: optuna.Trial):
        """Optuna objective function"""

        params = self.get_hyperparameter_space(trial)

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.seed
        )
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self.X_train, self.y_train)
        ):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]

            fold_model = self.create_model(params)
            fold_model = self.train_model(
                fold_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold
            )

            y_val_pred = fold_model.predict(X_val_fold)

            from evaluation.metrics import calculate_gmean

            fold_gmean = calculate_gmean(y_val_fold, y_val_pred)
            cv_scores.append(fold_gmean)

        mean_cv_score = np.mean(cv_scores)
        return mean_cv_score

    def optimize_hyperparameters(self):
        """Run Optuna hyperparameter optimization"""
        print("\nStarting hyperparameter optimization...")
        print(f"  Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
        print(f"  Device: {'GPU' if self.use_gpu else 'CPU'}")

        start_time = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=False)

        self.tuning_time = time.time() - start_time
        self.best_params = study.best_params

        print(f"\n✓ Optimization complete! Time: {self.tuning_time:.2f}s")
        print(f"  Best CV G-Mean: {study.best_value:.4f}")
        print("  Best parameters:")
        for param, value in self.best_params.items():
            print(f"    {param}: {value}")

        return self.best_params

    def train_final_model(self):
        """Train final model"""
        print("\nTraining final model...")

        start_time = time.time()

        if self.best_params is None:
            raise ValueError("Best hyperparameters not found. Run optimization first.")

        self.best_model = self.create_model(self.best_params)

        val_size = int(0.1 * len(self.X_train))
        indices = np.random.RandomState(self.seed).permutation(len(self.X_train))

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_final = self.X_train[train_indices]
        y_train_final = self.y_train[train_indices]
        X_val_final = self.X_train[val_indices]
        y_val_final = self.y_train[val_indices]

        self.best_model = self.train_model(
            self.best_model,
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
        )

        self.training_time = time.time() - start_time
        print(f"✓ Training complete! Time: {self.training_time:.2f}s")

    def evaluate(self) -> dict:
        """Evaluate final model on test set"""
        print("\nEvaluating on test set...")

        start_time = time.time()

        if self.best_model is None:
            raise ValueError("Best model is not trained yet.")

        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)

        self.prediction_time = time.time() - start_time

        y_true = self.y_test

        metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba, self.n_classes)

        results = {
            "dataset": self.dataset_name,
            "model": self.model_type,
            **metrics,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "tuning_time": self.tuning_time,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "total_time": self.tuning_time + self.training_time + self.prediction_time,
            "best_params": self.best_params,
            "device": "GPU" if self.use_gpu else "CPU",
        }

        print(f"\n{'=' * 80}")
        print(f"RESULTS FOR {self.dataset_name}")
        print(f"{'=' * 80}")
        print(f"Accuracy:        {results['accuracy']:.4f}")
        print(f"AUC OVO:         {results['auc_ovo']:.4f}")
        print(f"G-Mean:          {results['gmean']:.4f}")
        print(f"Cross-Entropy:   {results['cross_entropy']:.4f}")
        print(f"{'=' * 80}")

        return results

    def run_complete_experiment(self) -> dict:
        """Run the complete experiment pipeline"""
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {self.dataset_name}")
        print(f"{'=' * 80}")

        self.optimize_hyperparameters()
        self.train_final_model()
        results = self.evaluate()

        return results
