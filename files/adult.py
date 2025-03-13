# Part 1: Decision Trees with Categorical Attributes
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def read_csv_1(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(columns=['fnlwgt'])
    return df

def num_rows(df):
    return len(df)

def column_names(df):
    return df.columns.tolist()

def missing_values(df):
    return df.isin(['?']).sum().sum()

def columns_with_missing_values(df):
    return df.columns[df.isin(['?']).any()].tolist()

def bachelors_masters_percentage(df):
    education_filter = df['education'].isin(['Bachelors', 'Masters'])
    percentage = (education_filter.sum() / len(df)) * 100
    return round(percentage, 1)

def data_frame_without_missing_values(df):
    return df[~df.isin(['?']).any(axis=1)].copy()

def one_hot_encoding(df):
    df_clean = data_frame_without_missing_values(df)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns[:-1]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df_clean[categorical_cols])
    
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
    
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    result_df = pd.concat([df_clean[numerical_cols], encoded_df], axis=1)
    
    return result_df

def label_encoding(df):
    df_clean = data_frame_without_missing_values(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean['class'])
    return pd.Series(y, index=df_clean.index)

def dt_predict(X_train, y_train):
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_train)
    return pd.Series(y_pred, index=X_train.index)

def dt_error_rate(y_pred, y_true):
    return (y_pred != y_true).mean()