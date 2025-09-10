"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Script for preprocessing data for machine learning tasks (specifically Project 1: Foundations of Classification Algorithms).

This script includes functions for:
- loading and validating raw datasets
- cleaning and imputing missing values
- encoding categorical features
- scaling and normalizing numerical features
- splitting data into training, validation, and test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

def load_data(file_path: Path | str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to preprocess a dataframe. Includes handling missing values, encoding categorical variables, and scaling numerical features."""
    # Create copy to avoid modifying original dataframe
    df = df.copy()

    # Identify feature types. Exclude id column from being included in transformations
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_columns = []
    if "Unnamed: 0" in numerical_cols:
        numerical_cols.remove("Unnamed: 0")
        id_columns.append("Unnamed: 0")

    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=160)), # number of features based on sqrt(n_samples)
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], sparse_threshold=0.0)

    # Fit and transform the data
    processed_array = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(data=processed_array, # type: ignore[arg-type]
                                columns=feature_names, index=df.index)

    # Add ID column(s) back (untouched)
    if id_columns:
        processed_df[id_columns] = df[id_columns]

    # Reorder columns so ID is first again
    processed_df = processed_df[id_columns + feature_names.tolist()]

    return processed_df

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataframe in training and test sets."""
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test
