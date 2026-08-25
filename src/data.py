# Script for loading and splitting data and extracting feature names

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath="data/processed/df_final_num_cat.csv", test_size=0.2, random_state=126):
    """
    Loads the processed LNP dataset, separates features and target, 
    and returns train/test splits.
    """
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=["encapsulation_efficiency"])
    y = df["encapsulation_efficiency"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def get_feature_names(filepath="data/processed/df_final_num_cat.csv"):
    """Returns the list of feature names used for training."""
    df = pd.read_csv(filepath)
    return list(df.drop(columns=["encapsulation_efficiency"]).columns)