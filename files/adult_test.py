import os
import pandas as pd
from typing import List

# Import all functions from adult.py
from adult import (
    read_csv_1, 
    num_rows, 
    column_names, 
    missing_values, 
    columns_with_missing_values,
    bachelors_masters_percentage, 
    data_frame_without_missing_values, 
    one_hot_encoding, 
    label_encoding, 
    dt_predict, 
    dt_error_rate
)

class TestDataLoading:
    def __init__(self, data_file: str):
        self.data_file = data_file

    def test_read_csv_1(self) -> None:
        print("\n--- Testing read_csv_1 ---")
        df = read_csv_1(self.data_file)
        print(f"Shape of dataframe: {df.shape}")
        print(f"First 5 rows:")
        print(df.head())

        if "fnlwgt" in df.columns:
            print("WARNING: fnlwgt column was not dropped")
        else:
            print("fnlwgt column was correctly dropped")

    def test_num_rows(self) -> None:
        print("\n--- Testing num_rows ---")
        df = read_csv_1(self.data_file)
        count = num_rows(df)
        print(f"Number of rows in the dataframe: {count}")

    def test_column_names(self) -> None:
        print("\n--- Testing column_names ---")
        df = read_csv_1(self.data_file)
        cols = column_names(df)
        print(f"Column names ({len(cols)}):")
        for i, col in enumerate(cols, 1):
            print(f"  {i}. {col}")


class TestDataAnalysis:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = read_csv_1(data_file)

    def test_missing_values(self) -> None:
        print("\n--- Testing missing_values ---")
        missing = missing_values(self.df)
        print(f"Total number of missing values: {missing}")

        missing_by_col = self.df.isna().sum()
        print("Missing values by column:")
        for col, count in missing_by_col[missing_by_col > 0].items():
            print(f"  {col}: {count}")

    def test_columns_with_missing_values(self) -> None:
        print("\n--- Testing columns_with_missing_values ---")
        cols_missing = columns_with_missing_values(self.df)
        print(f"Columns with missing values ({len(cols_missing)}):")
        for col in cols_missing:
            count = self.df[col].isna().sum()
            print(f"  {col}: {count} missing values")

    def test_bachelors_masters_percentage(self) -> None:
        print("\n--- Testing bachelors_masters_percentage ---")
        percentage = bachelors_masters_percentage(self.df)
        print(
            f"Percentage of instances with Bachelor's or Master's education: {percentage}%"
        )

        count = self.df["education"].isin(["Bachelors", "Masters"]).sum()
        manual_percentage = round((count / len(self.df)) * 100, 1)
        print(
            f"Verification: {count} out of {len(self.df)} instances ({manual_percentage}%)"
        )


class TestDataPreprocessing:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = read_csv_1(data_file)

    def test_data_frame_without_missing_values(self) -> None:
        print("\n--- Testing data_frame_without_missing_values ---")
        df_clean = data_frame_without_missing_values(self.df)
        original_count = len(self.df)
        clean_count = len(df_clean)
        removed = original_count - clean_count
        print(f"Original number of rows: {original_count}")
        print(f"Number of rows after removing missing values: {clean_count}")
        print(
            f"Number of rows removed: {removed} ({(removed / original_count) * 100:.2f}%)"
        )

        missing_after = df_clean.isna().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")

    def test_one_hot_encoding(self) -> None:
        print("\n--- Testing one_hot_encoding ---")
        df_clean = data_frame_without_missing_values(self.df)
        X_encoded = one_hot_encoding(df_clean)

        print(f"Shape before encoding (with class column): {df_clean.shape}")
        print(f"Shape after encoding (without class column): {X_encoded.shape}")
        print(
            f"Number of columns increased from {len(df_clean.columns) - 1} to {len(X_encoded.columns)}"
        )
        print("\nFirst 5 columns of encoded dataframe:")
        print(X_encoded.iloc[:, :5].head())

        if "class" in X_encoded.columns:
            print("WARNING: 'class' column was not excluded")
        else:
            print("'class' column was correctly excluded")

    def test_label_encoding(self) -> None:
        print("\n--- Testing label_encoding ---")
        df_clean = data_frame_without_missing_values(self.df)
        y_encoded = label_encoding(df_clean)

        print(f"Original class values: {df_clean['class'].unique()}")
        print(f"Encoded class values: {y_encoded.unique()}")
        print(f"Original class distribution:\n{df_clean['class'].value_counts()}")
        print(f"Encoded class distribution:\n{y_encoded.value_counts()}")


class TestModelBuilding:
    def __init__(self, data_file: str):
        self.data_file = data_file
        df = read_csv_1(data_file)
        self.df_clean = data_frame_without_missing_values(df)
        self.X_encoded = one_hot_encoding(self.df_clean)
        self.y_encoded = label_encoding(self.df_clean)

    def test_dt_predict(self) -> None:
        print("\n--- Testing dt_predict ---")
        predictions = dt_predict(self.X_encoded, self.y_encoded)

        print(f"Shape of input features: {self.X_encoded.shape}")
        print(f"Shape of target: {self.y_encoded.shape}")
        print(f"Shape of predictions: {predictions.shape}")
        print(f"Unique predicted values: {predictions.unique()}")
        print(f"Distribution of predictions:\n{predictions.value_counts()}")

    def test_dt_error_rate(self) -> None:
        print("\n--- Testing dt_error_rate ---")
        predictions = dt_predict(self.X_encoded, self.y_encoded)
        error_rate = dt_error_rate(predictions, self.y_encoded)

        print(f"Decision tree error rate: {error_rate:.4f}")
        print(f"Decision tree accuracy: {1 - error_rate:.4f}")

        incorrect = (predictions != self.y_encoded).sum()
        total = len(self.y_encoded)
        manual_error_rate = incorrect / total
        print(
            f"Verification: {incorrect} incorrect predictions out of {total} ({manual_error_rate:.4f} error rate)"
        )


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)


def run_tests() -> None:
    data_file = "/Users/dan/Documents/Projects/DM/data/adult.csv"
    if not os.path.exists(data_file):
        print(f"Error: Dataset not found at {data_file}")
        print("Please ensure the adult.csv file is in the data folder.")
        return

    print_separator("TASK 1 - DECISION TREES WITH CATEGORICAL ATTRIBUTES")

    # Test data loading
    print_separator("DATA LOADING")
    data_loading_tests = TestDataLoading(data_file)
    data_loading_tests.test_read_csv_1()
    data_loading_tests.test_num_rows()
    data_loading_tests.test_column_names()

    # Test data analysis
    print_separator("DATA ANALYSIS")
    data_analysis_tests = TestDataAnalysis(data_file)
    data_analysis_tests.test_missing_values()
    data_analysis_tests.test_columns_with_missing_values()
    data_analysis_tests.test_bachelors_masters_percentage()

    # Test data preprocessing
    print_separator("DATA PREPROCESSING")
    data_preprocessing_tests = TestDataPreprocessing(data_file)
    data_preprocessing_tests.test_data_frame_without_missing_values()
    data_preprocessing_tests.test_one_hot_encoding()
    data_preprocessing_tests.test_label_encoding()

    # Test model building
    print_separator("MODEL BUILDING")
    model_building_tests = TestModelBuilding(data_file)
    model_building_tests.test_dt_predict()
    model_building_tests.test_dt_error_rate()

    print_separator("ALL TESTS COMPLETED")


if __name__ == "__main__":
    run_tests()
