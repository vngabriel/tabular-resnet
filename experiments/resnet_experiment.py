import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from evaluation.metrics import calculate_all_metrics, calculate_gmean
from models.resnet import TabularResNet


class TabularResNetExperiment:
    """Experiment pipeline for TabularResNet

    Expects preprocessed data (numpy arrays) from DataProcessor.
    Focuses only on model training and evaluation.
    """

    def __init__(
        self,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int,
        n_trials: int = 50,
        cv_folds: int = 10,
        seed: int = 123,
        device: str | None = None,
    ):
        """
        Initialize experiment

        Args:
            dataset_name: Name of the dataset
            X_train: Preprocessed training features (numpy array)
            y_train: Encoded training labels (numpy array)
            X_test: Preprocessed test features (numpy array)
            y_test: Encoded test labels (numpy array)
            n_classes: Number of classes
            n_trials: Number of Optuna trials for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            seed: Random seed
            device: Device to use ('cuda' or 'cpu')
        """
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.seed = seed

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Dataset info
        self.n_features = X_train.shape[1]
        self.n_samples_train = len(X_train)
        self.n_samples_test = len(X_test)

        print(f"Dataset: {dataset_name}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {self.n_features}, Classes: {self.n_classes}")

        # Results storage
        self.best_params = None
        self.best_model = None

        # Timing
        self.tuning_time = 0
        self.training_time = 0
        self.prediction_time = 0

        # Convert to tensors and move to GPU
        self.X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        self.y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        self.X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        self.y_test_tensor = torch.LongTensor(self.y_test).to(self.device)

        print(f"✓ Data loaded to {self.device}")

    def create_model(
        self, trial: optuna.Trial | None = None, params: dict | None = None
    ):
        """Create ResNet model with given hyperparameters"""

        if params is None and trial is not None:
            # Sample hyperparameters for Optuna
            params = {
                "d": trial.suggest_categorical("d", [64, 128, 256, 512]),
                "d_hidden_factor": trial.suggest_float("d_hidden_factor", 1.0, 5.0),
                "n_layers": trial.suggest_int("n_layers", 1, 10),
                "hidden_dropout": trial.suggest_float("hidden_dropout", 0.0, 0.5),
                "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256]
                ),
                "epochs": trial.suggest_int("epochs", 30, 200),
            }

        if params is None:
            raise ValueError("Either params or trial must be provided")

        model = TabularResNet(
            d_in=self.n_features,
            d=params["d"],
            d_hidden_factor=params["d_hidden_factor"],
            n_layers=params["n_layers"],
            hidden_dropout=params["hidden_dropout"],
            residual_dropout=params["residual_dropout"],
            d_out=self.n_classes,
        )

        return model, params

    def train_model(
        self,
        model,
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        params: dict,
        verbose: bool = False,
    ):
        """Train model with given data"""

        model = model.to(self.device)
        # Data is already on GPU from initialization - no need to transfer

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        batch_size = params["batch_size"]
        n_samples = len(X_train_tensor)
        n_batches = max(1, n_samples // batch_size)

        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(params["epochs"]):
            model.train()
            epoch_loss = 0

            # Shuffle data
            indices = torch.randperm(n_samples)

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"    Epoch {epoch + 1}/{params['epochs']}: "
                    f"Train Loss = {epoch_loss / n_batches:.4f}, Val Loss = {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch + 1}")
                    break

        return model

    def objective(self, trial: optuna.Trial):
        """Optuna objective function for hyperparameter optimization"""

        _, params = self.create_model(trial)

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.seed
        )
        cv_scores = []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            # Split data - index directly on GPU tensors
            X_train_fold = self.X_train_tensor[train_idx]
            y_train_fold = self.y_train_tensor[train_idx]
            X_val_fold = self.X_train_tensor[val_idx]
            y_val_fold = self.y_train_tensor[val_idx]

            # Create fresh model for this fold
            fold_model, _ = self.create_model(params=params)

            # Train
            fold_model = self.train_model(
                fold_model,
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                params,
                verbose=False,
            )

            # Evaluate (using G-Mean as optimization metric)
            fold_model.eval()
            with torch.no_grad():
                val_outputs = fold_model(X_val_fold)
                val_proba = torch.softmax(val_outputs, dim=1).cpu().numpy()

            y_val_true = y_val_fold.cpu().numpy()
            y_val_pred = val_proba.argmax(axis=1)

            # Calculate G-Mean
            fold_gmean = calculate_gmean(y_val_true, y_val_pred)
            cv_scores.append(fold_gmean)

        mean_cv_score = np.mean(cv_scores)
        return mean_cv_score

    def optimize_hyperparameters(self):
        """Run Optuna hyperparameter optimization"""
        print("\nStarting hyperparameter optimization...")
        print(f"  Trials: {self.n_trials}, CV Folds: {self.cv_folds}")

        start_time = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        study.optimize(self.objective, n_trials=self.n_trials)  # type: ignore

        self.tuning_time = time.time() - start_time
        self.best_params = study.best_params

        print(f"\n✓ Optimization complete! Time: {self.tuning_time:.2f}s")
        print(f"  Best CV G-Mean: {study.best_value:.4f}")
        print("  Best parameters:")
        for param, value in self.best_params.items():
            print(f"    {param}: {value}")

        return self.best_params

    def train_final_model(self):
        """Train final model with best hyperparameters on full training set"""
        print("\nTraining final model...")

        start_time = time.time()

        if self.best_params is None:
            raise ValueError("Best hyperparameters not found. Run optimization first.")

        # Create model with best params
        self.best_model, _ = self.create_model(params=self.best_params)

        # Use a small validation set for early stopping
        val_size = int(0.1 * len(self.X_train_tensor))
        indices = torch.randperm(len(self.X_train_tensor))

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_final = self.X_train_tensor[train_indices]
        y_train_final = self.y_train_tensor[train_indices]
        X_val_final = self.X_train_tensor[val_indices]
        y_val_final = self.y_train_tensor[val_indices]

        # Train
        self.best_model = self.train_model(
            self.best_model,
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            self.best_params,
            verbose=True,
        )

        self.training_time = time.time() - start_time
        print(f"✓ Training complete! Time: {self.training_time:.2f}s")

    def evaluate(self) -> dict:
        """Evaluate final model on test set"""
        print("\nEvaluating on test set...")

        start_time = time.time()

        if self.best_model is None:
            raise ValueError("Best model is not trained yet.")

        self.best_model.eval()
        with torch.no_grad():
            test_outputs = self.best_model(self.X_test_tensor)
            y_pred_proba = torch.softmax(test_outputs, dim=1).cpu().numpy()

        self.prediction_time = time.time() - start_time

        y_pred = y_pred_proba.argmax(axis=1)
        y_true = self.y_test

        # Calculate all metrics using the metrics module
        metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba, self.n_classes)

        # Create results dictionary
        results = {
            "dataset": self.dataset_name,
            "model": "resnet",
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
        }

        # Print results
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

        # Step 1: Optimize hyperparameters
        self.optimize_hyperparameters()

        # Step 2: Train final model
        self.train_final_model()

        # Step 3: Evaluate
        results = self.evaluate()

        return results
