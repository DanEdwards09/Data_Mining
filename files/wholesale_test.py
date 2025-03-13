import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time
from sklearn.metrics import silhouette_score


# Import all functions from wholesale_customers.py
from wholesale_customers import (
    read_csv_2,
    summary_statistics,
    standardize,
    kmeans,
    kmeans_plus,
    agglomerative,
    clustering_score,
    cluster_evaluation,
    best_clustering_score,
    scatter_plots
)
class TestDataLoading:
    def __init__(self, data_file: str):
        self.data_file = data_file

    def test_read_csv_2(self) -> pd.DataFrame:
        print("\n--- Testing read_csv_2 ---")
        df = read_csv_2(self.data_file)
        print(f"Shape of dataframe: {df.shape}")
        print(f"Columns in dataframe: {df.columns.tolist()}")
        print(f"First 5 rows:")
        print(df.head())

        # Check if Channel and Region columns were dropped
        if "Channel" in df.columns or "Region" in df.columns:
            print("WARNING: Channel and/or Region columns were not dropped")
        else:
            print("Channel and Region columns were correctly dropped")

        return df


class TestSummaryStatistics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_summary_statistics(self) -> None:
        print("\n--- Testing summary_statistics ---")
        stats = summary_statistics(self.df)
        print("Summary statistics shape:", stats.shape)
        print("Summary statistics columns:", stats.columns.tolist())
        print("Summary statistics sample:")
        print(stats)

        # Verify stats are correct
        for col in ["mean", "std"]:
            # Check if values are integers
            if not all(isinstance(val, (int, np.integer)) for val in stats[col]):
                print(f"WARNING: {col} values are not all integers")
            else:
                print(f"{col} values are correctly rounded to integers")

        # Manual verification of some values
        for attr in self.df.columns:
            actual_mean = self.df[attr].mean()
            actual_std = self.df[attr].std()
            actual_min = self.df[attr].min()
            actual_max = self.df[attr].max()

            rounded_mean = round(actual_mean)
            rounded_std = round(actual_std)

            if abs(stats.loc[attr, "mean"] - rounded_mean) > 0:
                print(
                    f"WARNING: Inconsistent mean for {attr}: expected {rounded_mean}, got {stats.loc[attr, 'mean']}"
                )

            if abs(stats.loc[attr, "std"] - rounded_std) > 0:
                print(
                    f"WARNING: Inconsistent std for {attr}: expected {rounded_std}, got {stats.loc[attr, 'std']}"
                )

            if stats.loc[attr, "min"] != actual_min:
                print(
                    f"WARNING: Inconsistent min for {attr}: expected {actual_min}, got {stats.loc[attr, 'min']}"
                )

            if stats.loc[attr, "max"] != actual_max:
                print(
                    f"WARNING: Inconsistent max for {attr}: expected {actual_max}, got {stats.loc[attr, 'max']}"
                )


class TestDataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_standardize(self) -> pd.DataFrame:
        print("\n--- Testing standardize ---")
        df_std = standardize(self.df)
        print("Original dataframe shape:", self.df.shape)
        print("Standardized dataframe shape:", df_std.shape)

        print("\nStandardized data sample:")
        print(df_std.head())

        print("\nStandardized data statistics:")
        print(f"Mean (should be close to 0):\n{df_std.mean()}")
        print(f"Standard deviation (should be close to 1):\n{df_std.std()}")

        # Verify standardization formula
        for col in self.df.columns:
            expected = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            diff = (df_std[col] - expected).abs().mean()
            if diff > 1e-10:
                print(
                    f"WARNING: Inconsistent standardization for {col}, mean difference: {diff}"
                )
            else:
                print(f"Standardization correct for {col}")

        return df_std


class TestClustering:
    def __init__(self, df: pd.DataFrame, df_std: pd.DataFrame):
        self.df = df
        self.df_std = df_std

    def test_kmeans(self) -> None:
        print("\n--- Testing kmeans ---")
        for k in [3, 5]:
            print(f"\nRunning k-means with k={k}")
            y = kmeans(self.df_std, k)
            print(f"Shape of cluster assignments: {y.shape}")
            print(f"Unique cluster labels: {sorted(y.unique())}")
            print(f"Cluster distribution: {y.value_counts().to_dict()}")

            # Calculate silhouette score
            score = silhouette_score(self.df_std, y)
            print(f"Silhouette score: {score:.4f}")

    def test_kmeans_plus(self) -> None:
        print("\n--- Testing kmeans_plus ---")
        for k in [3, 5]:
            print(f"\nRunning k-means++ with k={k}")
            y = kmeans_plus(self.df_std, k)
            print(f"Shape of cluster assignments: {y.shape}")
            print(f"Unique cluster labels: {sorted(y.unique())}")
            print(f"Cluster distribution: {y.value_counts().to_dict()}")

            # Calculate silhouette score
            score = silhouette_score(self.df_std, y)
            print(f"Silhouette score: {score:.4f}")

    def test_agglomerative(self) -> None:
        print("\n--- Testing agglomerative ---")
        for k in [3, 5]:
            print(f"\nRunning agglomerative clustering with k={k}")
            y = agglomerative(self.df_std, k)
            print(f"Shape of cluster assignments: {y.shape}")
            print(f"Unique cluster labels: {sorted(y.unique())}")
            print(f"Cluster distribution: {y.value_counts().to_dict()}")

            # Calculate silhouette score
            score = silhouette_score(self.df_std, y)
            print(f"Silhouette score: {score:.4f}")


class TestClusterEvaluation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_clustering_score(self) -> None:
        print("\n--- Testing clustering_score ---")
        # Test with a small sample
        small_df = self.df.sample(n=min(50, len(self.df)), random_state=42)

        # Get cluster assignments using kmeans
        y_kmeans = kmeans(small_df, 3)

        # Calculate silhouette score
        score = clustering_score(small_df, y_kmeans)
        expected_score = silhouette_score(small_df, y_kmeans)

        print(f"Calculated silhouette score: {score:.4f}")
        print(f"Expected silhouette score: {expected_score:.4f}")

        if abs(score - expected_score) < 1e-10:
            print("Silhouette score calculation is correct")
        else:
            print("WARNING: Inconsistent silhouette score calculation")

    def test_cluster_evaluation(self) -> None:
        print("\n--- Testing cluster_evaluation ---")

        # Use a smaller sample to speed up testing
        small_df = self.df.sample(n=min(50, len(self.df)), random_state=42)

        start_time = time.time()
        results = cluster_evaluation(small_df)
        elapsed_time = time.time() - start_time

        print(f"Cluster evaluation completed in {elapsed_time:.2f} seconds")
        print(f"Shape of results: {results.shape}")
        print(f"Columns: {results.columns.tolist()}")

        # Verify there are results for all combinations
        k_values = [3, 5, 10]
        expected_rows = (
            len(k_values) * 2 * (1 + 10)
        )  # 2 data types, 1 agg + 10 kmeans runs

        print(f"Expected number of rows: {expected_rows}")
        print(f"Actual number of rows: {len(results)}")

        if len(results) != expected_rows:
            print("WARNING: Unexpected number of result rows")

        # Check distribution of algorithms and k values
        print("\nAlgorithm distribution:")
        print(results["Algorithm"].value_counts())

        print("\nk value distribution:")
        print(results["k"].value_counts())

        print("\nData type distribution:")
        print(results["data"].value_counts())

        # Show summary of silhouette scores
        print("\nSilhouette score summary:")
        for algorithm in results["Algorithm"].unique():
            for data_type in results["data"].unique():
                subset = results[
                    (results["Algorithm"] == algorithm) & (results["data"] == data_type)
                ]
                print(f"{algorithm} on {data_type} data:")
                print(f"  Min score: {subset['Silhouette Score'].min():.4f}")
                print(f"  Max score: {subset['Silhouette Score'].max():.4f}")
                print(f"  Mean score: {subset['Silhouette Score'].mean():.4f}")

        return results

    def test_best_clustering_score(self, results: pd.DataFrame) -> None:
        print("\n--- Testing best_clustering_score ---")
        best_score = best_clustering_score(results)
        expected_best = results["Silhouette Score"].max()

        print(f"Calculated best silhouette score: {best_score:.4f}")
        print(f"Expected best silhouette score: {expected_best:.4f}")

        if abs(best_score - expected_best) < 1e-10:
            print("Best silhouette score calculation is correct")
        else:
            print("WARNING: Inconsistent best silhouette score calculation")

        # Show the configuration that got the best score
        best_config = results[results["Silhouette Score"] == best_score].iloc[0]
        print("\nBest configuration:")
        print(f"  Algorithm: {best_config['Algorithm']}")
        print(f"  Data type: {best_config['data']}")
        print(f"  k value: {best_config['k']}")


class TestVisualization:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_scatter_plots(self) -> None:
        print("\n--- Testing scatter_plots ---")

        # Use a smaller sample to speed up testing
        small_df = self.df.sample(n=min(50, len(self.df)), random_state=42)

        # Make a plots folder if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Save current working directory to restore it later
        current_dir = os.getcwd()
        os.chdir("plots")

        try:
            scatter_plots(small_df)

            # Check if plots were created
            plot_files = [
                f
                for f in os.listdir()
                if f.startswith("cluster_plot_") and f.endswith(".png")
            ]

            print(f"Number of scatter plots created: {len(plot_files)}")

            # Calculate expected number of plots
            num_attributes = len(small_df.columns)
            expected_plots = num_attributes * (num_attributes - 1) // 2

            if len(plot_files) == expected_plots:
                print(
                    f"Correct number of plots created ({expected_plots} pairs for {num_attributes} attributes)"
                )
            else:
                print(
                    f"WARNING: Expected {expected_plots} plots, but found {len(plot_files)}"
                )

            # Print the list of plot files
            if plot_files:
                print("\nPlot files created:")
                for file in plot_files[:5]:  # Limit to first 5 to avoid clutter
                    print(f"  {file}")
                if len(plot_files) > 5:
                    print(f"  ... and {len(plot_files) - 5} more")

        finally:
            # Restore original working directory
            os.chdir(current_dir)


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)


def run_tests() -> None:
    data_file = "/Users/dan/Documents/Projects/DM/data/wholesale_customers.csv"
    if not os.path.exists(data_file):
        print(f"Error: Dataset not found at {data_file}")
        print("Please ensure the wholesale_customers.csv file is in the data folder.")
        return

    print_separator("TASK 2 - CLUSTER ANALYSIS")

    # Test data loading
    print_separator("DATA LOADING")
    data_loading_tests = TestDataLoading(data_file)
    df = data_loading_tests.test_read_csv_2()

    # Test summary statistics
    print_separator("SUMMARY STATISTICS")
    summary_tests = TestSummaryStatistics(df)
    summary_tests.test_summary_statistics()

    # Test data preprocessing
    print_separator("DATA PREPROCESSING")
    data_preprocessing_tests = TestDataPreprocessing(df)
    df_std = data_preprocessing_tests.test_standardize()

    # Test clustering algorithms
    print_separator("CLUSTERING ALGORITHMS")
    clustering_tests = TestClustering(df, df_std)
    clustering_tests.test_kmeans()
    clustering_tests.test_kmeans_plus()
    clustering_tests.test_agglomerative()

    # Test cluster evaluation
    print_separator("CLUSTER EVALUATION")
    eval_tests = TestClusterEvaluation(df)
    eval_tests.test_clustering_score()
    results = eval_tests.test_cluster_evaluation()
    eval_tests.test_best_clustering_score(results)

    # Test visualization
    print_separator("VISUALIZATION")
    vis_tests = TestVisualization(df)
    vis_tests.test_scatter_plots()

    print_separator("ALL TESTS COMPLETED")


if __name__ == "__main__":
    run_tests()
