import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


def load_and_preprocess(file_path: str):
    """
    Load the banking dataset and preprocess it.
    """
    data = pd.read_csv(file_path)

    # Drop unwanted columns
    cols = set(data.columns) - {'Location ID', 'BRId', 'GenderId', 'IAId'}
    data1 = data[list(cols)]

    return data1


def summarize_data(df: pd.DataFrame):
    """
    Return basic info and statistics of the dataset.
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "nulls": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
        "description": df.describe().T
    }
    return summary


if __name__ == "__main__":
    file_path = "Banking.csv"
    df = load_and_preprocess(file_path)
    summary = summarize_data(df)

    print("Shape:", summary["shape"])
    print("Columns:", summary["columns"])
    print("Null counts:", summary["nulls"])
    print("Unique counts:", summary["unique_values"])
    print("\nDescription:\n", summary["description"])
