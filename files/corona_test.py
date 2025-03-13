import os
import pandas as pd
import numpy as np
from typing import List, Set, Dict
import time
from collections import Counter

# Import all functions from coronavirus_tweets.py
from coronavirus_tweets import (
    read_csv_3,
    get_sentiments,
    second_most_popular_sentiment,
    date_most_popular_tweets,
    lower_case,
    remove_non_alphabetic_chars,
    remove_multiple_consecutive_whitespaces,
    tokenize,
    count_words_with_repetitions,
    count_words_without_repetitions,
    frequent_words,
    remove_stop_words,
    stemming,
    mnb_predict,
    mnb_accuracy
)

class TestDataLoading:
    def __init__(self, data_file: str):
        self.data_file = data_file

    def test_read_csv_3(self) -> pd.DataFrame:
        print("\n--- Testing read_csv_3 ---")
        df = read_csv_3(self.data_file)
        print(f"Shape of dataframe: {df.shape}")
        print(f"Columns in dataframe: {df.columns.tolist()}")
        print(f"First 5 rows:")
        for col in df.columns:
            print(f"\n{col}:")
            if col == "OriginalTweet":
                # For OriginalTweet, just show the first 80 chars
                for tweet in df[col].head():
                    print(f"  {str(tweet)[:80]}...")
            else:
                print(f"  {df[col].head().tolist()}")

        return df


class TestSentimentAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_get_sentiments(self) -> None:
        print("\n--- Testing get_sentiments ---")
        sentiments = get_sentiments(self.df)
        print(f"Possible sentiments ({len(sentiments)}):")
        for i, sentiment in enumerate(sentiments, 1):
            print(f"  {i}. {sentiment}")

        # Verify against direct count
        unique_sentiments = set(self.df["Sentiment"].dropna().unique())
        if set(sentiments) == unique_sentiments:
            print("All sentiments correctly identified")
        else:
            print("WARNING: Sentiment lists don't match")
            missing = unique_sentiments - set(sentiments)
            extra = set(sentiments) - unique_sentiments
            if missing:
                print(f"Missing sentiments: {missing}")
            if extra:
                print(f"Extra sentiments: {extra}")

    def test_second_most_popular_sentiment(self) -> None:
        print("\n--- Testing second_most_popular_sentiment ---")
        sentiment = second_most_popular_sentiment(self.df)
        print(f"Second most popular sentiment: {sentiment}")

        # Verify against direct count
        sentiment_counts = self.df["Sentiment"].value_counts()
        print("\nTop 3 sentiments by frequency:")
        for i, (s, count) in enumerate(sentiment_counts.head(3).items(), 1):
            print(f"  {i}. {s}: {count} tweets")

        if len(sentiment_counts) >= 2 and sentiment == sentiment_counts.index[1]:
            print("Second most popular sentiment correctly identified")
        else:
            print(f"WARNING: Expected {sentiment_counts.index[1]}, got {sentiment}")

    def test_date_most_popular_tweets(self) -> None:
        print("\n--- Testing date_most_popular_tweets ---")
        date = date_most_popular_tweets(self.df)
        print(f"Date with most extremely positive tweets: {date}")

        # Verify against direct count
        extremely_positive = self.df[self.df["Sentiment"] == "Extremely Positive"]
        if not extremely_positive.empty:
            date_counts = extremely_positive["TweetAt"].value_counts()
            print("\nTop 3 dates by extremely positive tweet frequency:")
            for i, (d, count) in enumerate(date_counts.head(3).items(), 1):
                print(f"  {i}. {d}: {count} tweets")

            if date == str(date_counts.idxmax()):
                print("Date with most extremely positive tweets correctly identified")
            else:
                print(f"WARNING: Expected {date_counts.idxmax()}, got {date}")
        else:
            print("No 'Extremely Positive' tweets found in the dataset")


class TestTextPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()

    def test_lower_case(self) -> pd.DataFrame:
        print("\n--- Testing lower_case ---")
        df = lower_case(self.df)

        # Check a few examples
        print("Original vs. Lowercase examples:")
        for i, (orig, lower) in enumerate(
            zip(self.original_df["OriginalTweet"].head(3), df["OriginalTweet"].head(3)),
            1,
        ):
            print(f"\nExample {i}:")
            print(f"  Original: {orig[:80]}...")
            print(f"  Lowercase: {lower[:80]}...")

        # Check if all are lowercase
        has_uppercase = df["OriginalTweet"].str.contains(r"[A-Z]").any()
        if not has_uppercase:
            print("\nAll tweets successfully converted to lowercase")
        else:
            print("\nWARNING: Some tweets still contain uppercase characters")

        return df

    def test_remove_non_alphabetic_chars(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Testing remove_non_alphabetic_chars ---")
        df_cleaned = remove_non_alphabetic_chars(df)

        # Check a few examples
        print("Before vs. After removing non-alphabetic characters:")
        for i, (before, after) in enumerate(
            zip(df["OriginalTweet"].head(3), df_cleaned["OriginalTweet"].head(3)), 1
        ):
            print(f"\nExample {i}:")
            print(f"  Before: {before[:80]}...")
            print(f"  After: {after[:80]}...")

        # Check if any non-alphabetic chars remain (except whitespace)
        has_non_alpha = df_cleaned["OriginalTweet"].str.contains(r"[^a-zA-Z\s]").any()
        if not has_non_alpha:
            print("\nAll non-alphabetic characters successfully removed")
        else:
            print("\nWARNING: Some non-alphabetic characters remain")

        return df_cleaned

    def test_remove_multiple_consecutive_whitespaces(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        print("\n--- Testing remove_multiple_consecutive_whitespaces ---")
        df_cleaned = remove_multiple_consecutive_whitespaces(df)

        # Check a few examples
        print("Before vs. After removing extra whitespaces:")
        for i, (before, after) in enumerate(
            zip(df["OriginalTweet"].head(3), df_cleaned["OriginalTweet"].head(3)), 1
        ):
            print(f"\nExample {i}:")
            print(f"  Before: {before[:80]}...")
            print(f"  After: {after[:80]}...")

        # Check if any multiple whitespaces remain
        has_multi_spaces = df_cleaned["OriginalTweet"].str.contains(r"\s\s+").any()
        if not has_multi_spaces:
            print("\nAll multiple consecutive whitespaces successfully removed")
        else:
            print("\nWARNING: Some multiple consecutive whitespaces remain")

        return df_cleaned

    def run_preprocessing_pipeline(self) -> pd.DataFrame:
        """Run the full preprocessing pipeline and return the preprocessed dataframe"""
        df_lower = self.test_lower_case()
        df_no_special = self.test_remove_non_alphabetic_chars(df_lower)
        df_clean = self.test_remove_multiple_consecutive_whitespaces(df_no_special)
        return df_clean


class TestTokenization:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_tokenize(self) -> pd.DataFrame:
        print("\n--- Testing tokenize ---")
        df_tokenized = tokenize(self.df)

        # Check if tokenized column was created
        if "tokenized" not in df_tokenized.columns:
            print("WARNING: 'tokenized' column not created")
            return df_tokenized

        # Check a few examples
        print("Original text vs. Tokenized examples:")
        for i, (text, tokens) in enumerate(
            zip(
                df_tokenized["OriginalTweet"].head(3), df_tokenized["tokenized"].head(3)
            ),
            1,
        ):
            print(f"\nExample {i}:")
            print(f"  Original: {text[:80]}...")
            print(f"  Tokenized: {tokens[:10]} {'...' if len(tokens) > 10 else ''}")

            # Verify tokens match the original text
            reconstructed = " ".join(tokens)
            if reconstructed == text:
                print("  ✓ Tokenization is correct")
            else:
                print("  ✗ Tokenization may be incorrect")
                print(f"  Reconstructed: {reconstructed[:80]}...")

        return df_tokenized

    def test_word_counts(self, df_tokenized: pd.DataFrame) -> None:
        print("\n--- Testing word counts ---")

        # Test count with repetitions
        total_words = count_words_with_repetitions(df_tokenized)
        print(f"Total words (with repetitions): {total_words}")

        # Verify with manual count
        manual_count = sum(len(tokens) for tokens in df_tokenized["tokenized"])
        if total_words == manual_count:
            print("✓ Total word count is correct")
        else:
            print(f"✗ Expected {manual_count}, got {total_words}")

        # Test count without repetitions
        distinct_words = count_words_without_repetitions(df_tokenized)
        print(f"Distinct words: {distinct_words}")

        # Verify with manual count
        all_words = set()
        for tokens in df_tokenized["tokenized"]:
            all_words.update(tokens)
        if distinct_words == len(all_words):
            print("✓ Distinct word count is correct")
        else:
            print(f"✗ Expected {len(all_words)}, got {distinct_words}")

    def test_frequent_words(self, df_tokenized: pd.DataFrame) -> None:
        print("\n--- Testing frequent_words ---")

        top_k = 10
        most_common = frequent_words(df_tokenized, top_k)

        print(f"Top {top_k} most frequent words:")
        for i, word in enumerate(most_common, 1):
            print(f"  {i}. {word}")

        # Verify with manual count
        all_tokens = []
        for tokens in df_tokenized["tokenized"]:
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)
        expected_top_words = [word for word, _ in counter.most_common(top_k)]

        if most_common == expected_top_words:
            print("✓ Most frequent words are correct")
        else:
            print("✗ Discrepancy in most frequent words")
            print(f"  Expected: {expected_top_words}")
            print(f"  Got: {most_common}")

    def test_remove_stop_words(self, df_tokenized: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Testing remove_stop_words ---")

        # Count tokens before
        total_before = sum(len(tokens) for tokens in df_tokenized["tokenized"])

        # Remove stop words
        df_no_stops = remove_stop_words(df_tokenized)

        # Count tokens after
        total_after = sum(len(tokens) for tokens in df_no_stops["tokenized"])

        print(f"Total tokens before: {total_before}")
        print(f"Total tokens after: {total_after}")
        print(
            f"Tokens removed: {total_before - total_after} ({(total_before - total_after) / total_before * 100:.1f}%)"
        )

        # Check a few examples
        print("\nBefore vs. After removing stop words:")
        for i, (before, after) in enumerate(
            zip(df_tokenized["tokenized"].head(3), df_no_stops["tokenized"].head(3)), 1
        ):
            print(f"\nExample {i}:")
            print(f"  Before: {before[:10]} {'...' if len(before) > 10 else ''}")
            print(f"  After: {after[:10]} {'...' if len(after) > 10 else ''}")

            # Check if short words and common stop words were removed
            short_words = [w for w in before if len(w) <= 2]
            short_words_after = [w for w in after if len(w) <= 2]

            common_stop_words = ["the", "and", "to", "of", "a", "in", "is", "that"]
            stops_before = [w for w in before if w in common_stop_words]
            stops_after = [w for w in after if w in common_stop_words]

            if len(short_words_after) == 0:
                print(f"  ✓ All {len(short_words)} short words removed")
            else:
                print(f"  ✗ {len(short_words_after)} short words remain")

            if len(stops_after) == 0:
                print(f"  ✓ All common stop words removed")
            else:
                print(f"  ✗ Some stop words remain: {stops_after}")

        return df_no_stops

    def test_stemming(self, df_tokenized: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Testing stemming ---")

        # Stem words
        df_stemmed = stemming(df_tokenized)

        # Check a few examples
        print("Before vs. After stemming:")
        for i, (before, after) in enumerate(
            zip(df_tokenized["tokenized"].head(3), df_stemmed["tokenized"].head(3)), 1
        ):
            print(f"\nExample {i}:")
            print(f"  Before: {before[:10]} {'...' if len(before) > 10 else ''}")
            print(f"  After: {after[:10]} {'...' if len(after) > 10 else ''}")

            # Simple check for common stemming patterns
            if len(before) == len(after):
                changes = sum(1 for w1, w2 in zip(before, after) if w1 != w2)
                print(f"  {changes} of {len(before)} words were stemmed")
            else:
                print("  WARNING: Different number of tokens before and after stemming")

        return df_stemmed

    def compare_frequent_words(
        self, df_original: pd.DataFrame, df_processed: pd.DataFrame
    ) -> None:
        print("\n--- Comparing frequent words before and after processing ---")

        top_k = 10
        print("Top 10 words before processing:")
        before = frequent_words(df_original, top_k)
        for i, word in enumerate(before, 1):
            print(f"  {i}. {word}")

        print("\nTop 10 words after processing (stop words removal and stemming):")
        after = frequent_words(df_processed, top_k)
        for i, word in enumerate(after, 1):
            print(f"  {i}. {word}")

        # Compare the two sets
        common = set(before) & set(after)
        print(f"\nWords appearing in both lists: {len(common)}")
        if common:
            print(f"Common words: {', '.join(common)}")

        print("\nObservations:")
        print("  - After processing, we typically see more meaningful content words")
        print("  - Stop words and common function words are removed")
        print(
            "  - Different word forms (e.g., 'running', 'runs') are reduced to their stems"
        )
        print("  - This helps focus on the actual content and reduces dimensionality")


class TestClassification:
    def __init__(self, df: pd.DataFrame, use_sample: bool = True):
        """
        Initialize the test class

        Args:
            df: The DataFrame to use
            use_sample: If True, use a sample of the data for faster testing
        """
        if use_sample:
            # Use a small sample for faster testing of mnb_predict
            self.df = df.sample(n=min(300, len(df)), random_state=42)
            print(f"Using a sample of {len(self.df)} tweets for classification tests")
        else:
            self.df = df
            print(f"Using all {len(self.df)} tweets for classification tests")

    def test_mnb_predict(self) -> np.ndarray:
        print("\n--- Testing mnb_predict ---")

        start_time = time.time()
        y_pred = mnb_predict(self.df)
        elapsed_time = time.time() - start_time

        print(f"Prediction completed in {elapsed_time:.2f} seconds")
        print(f"Shape of predictions: {y_pred.shape}")

        # Check prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        print("\nPrediction distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} ({count / len(y_pred) * 100:.1f}%)")

        # Check against actual distribution
        actual_labels = self.df["Sentiment"].values
        actual_unique, actual_counts = np.unique(actual_labels, return_counts=True)
        print("\nActual distribution:")
        for label, count in zip(actual_unique, actual_counts):
            print(f"  {label}: {count} ({count / len(actual_labels) * 100:.1f}%)")

        return y_pred

    def test_mnb_accuracy(self, y_pred: np.ndarray) -> None:
        print("\n--- Testing mnb_accuracy ---")

        y_true = self.df["Sentiment"].values
        accuracy = mnb_accuracy(y_pred, y_true)

        print(f"Calculated accuracy: {accuracy:.3f}")

        # Verify with manual calculation
        manual_accuracy = np.mean(y_pred == y_true)
        print(f"Manually calculated accuracy: {manual_accuracy:.6f}")

        if abs(accuracy - round(manual_accuracy, 3)) < 1e-10:
            print("✓ Accuracy calculation is correct")
        else:
            print(f"✗ Expected {round(manual_accuracy, 3)}, got {accuracy}")

        print("\nConfusion matrix (first 5 classes):")
        classes = sorted(set(np.concatenate([y_true, y_pred])))[:5]

        # Print header
        print("  Actual →")
        print("Pred ↓  " + " ".join(f"{c[:8]:9s}" for c in classes))

        # Print confusion matrix
        for pred_class in classes:
            row = []
            for true_class in classes:
                count = np.sum((y_pred == pred_class) & (y_true == true_class))
                row.append(f"{count:9d}")
            print(f"{pred_class[:8]:8s} " + " ".join(row))


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)


def run_tests(use_sample: bool = True) -> None:
    data_file = "/Users/dan/Documents/Projects/DM/data/coronavirus_tweets.csv"
    if not os.path.exists(data_file):
        print(f"Error: Dataset not found at {data_file}")
        print("Please ensure the coronavirus_tweets.csv file is in the data folder.")
        return

    print_separator("TASK 3 - TEXT MINING")

    # Test data loading
    print_separator("DATA LOADING")
    data_loading_tests = TestDataLoading(data_file)
    df = data_loading_tests.test_read_csv_3()

    # Test sentiment analysis
    print_separator("SENTIMENT ANALYSIS")
    sentiment_tests = TestSentimentAnalysis(df)
    sentiment_tests.test_get_sentiments()
    sentiment_tests.test_second_most_popular_sentiment()
    sentiment_tests.test_date_most_popular_tweets()

    # Test text preprocessing
    print_separator("TEXT PREPROCESSING")
    preprocessing_tests = TestTextPreprocessing(df)
    df_clean = preprocessing_tests.run_preprocessing_pipeline()

    # Test tokenization and word analysis
    print_separator("TOKENIZATION AND WORD ANALYSIS")
    tokenization_tests = TestTokenization(df_clean)
    df_tokenized = tokenization_tests.test_tokenize()
    tokenization_tests.test_word_counts(df_tokenized)
    tokenization_tests.test_frequent_words(df_tokenized)

    # Test stop word removal and stemming
    print_separator("STOP WORD REMOVAL AND STEMMING")
    df_no_stops = tokenization_tests.test_remove_stop_words(df_tokenized)
    df_stemmed = tokenization_tests.test_stemming(df_no_stops)
    tokenization_tests.compare_frequent_words(df_tokenized, df_stemmed)

    # Test classification
    print_separator("CLASSIFICATION")
    classification_tests = TestClassification(df, use_sample=use_sample)
    y_pred = classification_tests.test_mnb_predict()
    classification_tests.test_mnb_accuracy(y_pred)

    print_separator("ALL TESTS COMPLETED")


if __name__ == "__main__":
    # Set use_sample=False to use the full dataset (will be slower)
    run_tests(use_sample=True)
