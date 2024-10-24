# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Load the new CSV file
df = pd.read_csv('3_Data Encoding/OP_train.csv')


def create_dataset(n_sample=1000):
    '''
    Create an unevenly distributed sample dataset for multilabel classification.

    args:
    nsample: int, Number of samples to be created.

    return:
    X: pandas.DataFrame, feature vector dataframe with 10 features.
    y: pandas.DataFrame, target vector dataframe with 5 labels.
    '''
    X, y = make_classification(n_classes=5, class_sep=2,
                               weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y


def get_tail_label(df):
    """
    Get tail label columns of the given target dataframe.

    args:
    df: pandas.DataFrame, target label df whose tail label has to be identified.

    return:
    tail_label: list, a list containing column names of all the tail labels.
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def get_index(df):
    """
    Get the index of all tail label rows.

    args:
    df: pandas.DataFrame, target label df from which indices for tail labels are identified.

    return:
    index: list, a list containing index numbers of all the tail label rows.
    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)
        index = index.union(sub_index)
    return list(index)


def get_minority_instance(X, y):
    """
    Get minority dataframe containing all the tail labels.

    args:
    X: pandas.DataFrame, the feature vector dataframe.
    y: pandas.DataFrame, the target vector dataframe.

    return:
    X_sub: pandas.DataFrame, the feature vector minority dataframe.
    y_sub: pandas.DataFrame, the target vector minority dataframe.
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X, n_neighbors=5, algorithm='auto'):
    """
    Get index of 5 nearest neighbors for each instance using the specified algorithm.

    args:
    X: np.array, the feature vector for which to find nearest neighbors.
    n_neighbors: int, the number of nearest neighbors to find.
    algorithm: str, the algorithm to use ('kd_tree', 'ball_tree', 'brute', 'auto').

    return:
    indices: list of lists, index of 5 NN for each element in X.
    """
    print(X.shape)
    nbs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm=algorithm).fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Perform MLSMOTE (Multi-Label Synthetic Minority Over-sampling Technique).

    args:
    X: pandas.DataFrame, input vector DataFrame.
    y: pandas.DataFrame, feature vector dataframe.
    n_sample: int, number of newly generated samples.

    return:
    new_X: pandas.DataFrame, augmented feature vector data.
    target: pandas.DataFrame, augmented target vector data.
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


def undersampling(df, max_samples_per_class=1500):
    """
    Perform undersampling on the dataset, focusing on the least represented class.

    args:
    df: pandas.DataFrame, the dataset containing features and labels.
    max_samples_per_class: int, the maximum number of samples to retain for each class.

    return:
    df_balanced: pandas.DataFrame, the balanced dataset after undersampling.
    """
    labels = ['ARTHM', 'LE', 'None', 'RENT', 'TimeO']

    # Count samples for each label and sort by the number of instances
    label_counts = df[labels].sum().sort_values()

    # Create an empty DataFrame for the balanced data
    df_balanced = pd.DataFrame()

    # Perform undersampling from the least represented label to the most
    for label in label_counts.index:
        df_label = df[df[label] == 1]

        # Check how many samples of this label are already in df_balanced
        df_check = len(df_balanced[df_balanced[label] == 1]) if not df_balanced.empty else 0

        # If the number of samples for this label exceeds the max_samples_per_class, resample it
        if len(df_label) + df_check > max_samples_per_class:
            remaining_samples = max_samples_per_class - df_check
            if remaining_samples > 0:
                df_label = resample(df_label, replace=False, n_samples=remaining_samples, random_state=42)

        # Remove these rows from the original dataset
        df = df[~df.index.isin(df_label.index)]

        # Add the undersampled rows to the balanced DataFrame
        df_balanced = pd.concat([df_balanced, df_label])

    return df_balanced


if __name__ == '__main__':
    """
    Main function to apply MLSMOTE and perform resampling.
    """
    # Select columns for X (features) and y (labels)
    X = df.loc[:, [f'embedding_{i + 1}' for i in range(300)]]
    y = df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']]
    check = df.drop_duplicates()
    # Step 1: Get minority instances
    X_sub, y_sub = get_minority_instance(X, y)

    # Step 2: Apply MLSMOTE to augment the dataset
    X_res, y_res = MLSMOTE(X_sub, y_sub, 1000)

    # Step 3: Filter out 'ARTHM' and 'None' classes if they already have enough samples
    indices_to_keep = y_res[~((y_res['ARTHM'] == 1) | (y_res['None'] == 1))].index
    y_res = y_res.loc[indices_to_keep]
    X_res = X_res.loc[indices_to_keep]

    # Step 4: Combine original and resampled data
    print(df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    new_df = pd.concat([X_res, y_res], axis=1)
    print(new_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    result_df = pd.concat([df, new_df], ignore_index=True)
    print(result_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    # Step 5: Drop duplicates
    result_df = result_df.drop_duplicates()
    print(result_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    # Step 6: Apply undersampling
    new_df = undersampling(result_df)
    print(new_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    # Step 7: Save the final balanced dataset to CSV
    new_df.to_csv('3_Data Encoding/OP_train_final.csv')
