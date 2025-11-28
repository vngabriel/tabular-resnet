from pathlib import Path

import pandas as pd


class ResultsManager:
    """Manages experiment results, metrics CSV, and visualizations"""

    def __init__(self, save_dir: str = "./results", model_name: str = "model"):
        """
        Initialize results manager

        Args:
            save_dir: Directory to save all results
            model_name: Name of the model (e.g., 'resnet', 'xgboost', 'random_forest')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.model_name = model_name

        # Create subdirectories
        self.individual_dir = self.save_dir / "individual_results"
        self.individual_dir.mkdir(exist_ok=True)

        self.all_results = {}

    def save_dataset_result(
        self,
        dataset_name: str,
        metrics: dict[str, float],
        dataset_info: dict[str, int] | None = None,
        timings: dict[str, float] | None = None,
        hyperparameters: dict | None = None,
    ):
        """Save results for a single dataset

        Args:
            dataset_name: Name of the dataset
            metrics: Dictionary with metrics (accuracy, auc_ovo, gmean, cross_entropy)
            dataset_info: Optional dataset info (n_samples_train, n_samples_test, n_features, n_classes)
            timings: Optional timing info (preprocessing_time, tuning_time, training_time, etc.)
            hyperparameters: Optional best hyperparameters
        """
        result = {
            "dataset": dataset_name,
            "model": self.model_name,
            **metrics,
        }

        # Add optional info
        if dataset_info:
            result.update(dataset_info)
        if timings:
            result.update(timings)
        if hyperparameters:
            result["best_params"] = hyperparameters

        # Save individual result as CSV
        result_file = self.individual_dir / f"{dataset_name}_result.csv"

        # Flatten hyperparameters for CSV format
        result_flattened = result.copy()
        if "best_params" in result_flattened and isinstance(
            result_flattened["best_params"], dict
        ):
            for param, value in result_flattened["best_params"].items():
                result_flattened[f"hp_{param}"] = value
            del result_flattened["best_params"]

        # Convert to DataFrame and save
        df = pd.DataFrame([result_flattened])
        df.to_csv(result_file, index=False)

        # Store in memory
        self.all_results[dataset_name] = result

        print(f"✓ Saved results for {dataset_name}")

    def save_metrics_csv(self, filename: str | None = None):
        """Save metrics CSV for hypothesis testing

        This creates a CSV with one row per dataset containing all metrics.
        Perfect for sharing with colleagues and statistical analysis.

        Args:
            filename: Optional custom filename (default: {model_name}_metrics.csv)
        """
        if not self.all_results:
            print("No results to save. Run experiments first.")
            return

        # Create filename
        if filename is None:
            filename = f"{self.model_name}_metrics.csv"

        csv_path = self.save_dir / filename

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(self.all_results, orient="index")

        # Select only the important columns for hypothesis testing
        metric_columns = [
            "dataset",
            "model",
            "accuracy",
            "auc_ovo",
            "gmean",
            "cross_entropy",
        ]

        # Add dataset info if available
        optional_columns = [
            "n_samples_train",
            "n_samples_test",
            "n_features",
            "n_classes",
        ]
        for col in optional_columns:
            if col in df.columns:
                metric_columns.append(col)

        # Filter and sort
        df = df[metric_columns].sort_values("dataset").reset_index(drop=True)

        # Save CSV
        df.to_csv(csv_path, index=False)

        print(f"\n✓ Saved metrics CSV: {csv_path}")
        print(f"  Total datasets: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Print summary statistics
        print("\nMetrics Summary:")
        print(df[["accuracy", "auc_ovo", "gmean", "cross_entropy"]].describe())

        return df

    def print_summary(self):
        """Print summary of all results"""
        if not self.all_results:
            print("No results available.")
            return

        print("\n" + "=" * 80)
        print(f"RESULTS SUMMARY - {self.model_name.upper()}")
        print("=" * 80)
        print(f"Total datasets: {len(self.all_results)}")

        # Create DataFrame
        df = pd.DataFrame.from_dict(self.all_results, orient="index")

        # Print metrics statistics
        metric_cols = ["accuracy", "auc_ovo", "gmean", "cross_entropy"]
        print("\nMetrics Statistics:")
        print(df[metric_cols].describe())

        # Print top 5 datasets by accuracy
        print("\nTop 5 datasets by accuracy:")
        top5 = df.nlargest(5, "accuracy")[["dataset", "accuracy", "auc_ovo", "gmean"]]
        print(top5.to_string(index=False))

        # Print bottom 5 datasets by accuracy
        print("\nBottom 5 datasets by accuracy:")
        bottom5 = df.nsmallest(5, "accuracy")[
            ["dataset", "accuracy", "auc_ovo", "gmean"]
        ]
        print(bottom5.to_string(index=False))

        print("=" * 80)
