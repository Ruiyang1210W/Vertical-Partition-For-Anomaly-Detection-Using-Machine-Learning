import pandas as pd
import numpy as np

def load_and_preprocess(csv_path, label_column="Label", desired_features=100):
    # Load data
    df = pd.read_csv("C:/Users/wangr/OneDrive/Desktop/Senior project resourses/Vertical Partition/BalancedTestData.csv")
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], 0)


    # Separate features and Label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Normalize X
    mean_vals = np.mean(X, axis=0)
    std_vals = np.std(X, axis=0)
    std_vals[std_vals == 0] = 1
    X_normalized = (X - mean_vals) / std_vals

   # Pad features to match model input requirements
    original_feature = X_normalized.shape[1]
    if original_feature < desired_features:
        X_normalized = np.pad(X_normalized.values, ((0, 0), (0, desired_features - original_feature)), mode='constant')
    else:
        X_normalized = X_normalized.values

    return X_normalized, y.values, mean_vals, std_vals