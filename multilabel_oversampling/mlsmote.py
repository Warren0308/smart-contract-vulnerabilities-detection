# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
import os

from sklearn.utils import resample

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
# Load the new CSV file
df = pd.read_csv('3_Data Encoding/train_split.csv')
def create_dataset(n_sample=1000):
    '''
    Create a unevenly distributed sample data set multilabel
    classification using make_classification function

    args
    nsample: int, Number of sample to be created

    return
    X: pandas.DataFrame, feature vector dataframe with 10 features
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2,
                               weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y


def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe

    args
    df: pandas.DataFrame, target label df whose tail label has to identified

    return
    tail_label: list, a list containing column name of all the tail label
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
    give the index of all tail_label rows
    args
    df: pandas.DataFrame, target label df from which index for tail label has to identified

    return
    index: list, a list containing index number of all the tail label
    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)
        index = index.union(sub_index)
    return list(index)


def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels

    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe

    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    print(X.shape)
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def undersampling(df):
    # 选择标签列
    labels = ['ARTHM', 'LE', 'None', 'RENT', 'TimeO']

    # 计算每个标签的样本数
    label_counts = df[labels].sum().sort_values()

    print(label_counts)
    # 创建一个空的 DataFrame 来存储下采样后的数据
    df_balanced = pd.DataFrame()

    # 从少数类开始提取样本
    for label in label_counts.index:
        # 存储总提取样本数
        max_samples_per_class = 1500
        # 获取当前类别的样本
        df_label = df[df[label] == 1]
        if not df_balanced.empty:
            df_check = len(df_balanced[df_balanced[label] == 1])
        else:
            df_check = 0
        # 如果当前类别的样本数大于剩余可提取数，进行下采样
        if len(df_label) > max_samples_per_class and df_check<=1500:
            df_label = resample(df_label, replace=False, n_samples=max_samples_per_class-df_check, random_state=42)

        # 从原始数据中删除这些行
        df = df[~df.index.isin(df_label.index)]
        # 将当前类别的样本添加到结果 DataFrame
        df_balanced = pd.concat([df_balanced, df_label])
    return df_balanced

def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
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


if __name__ == '__main__':
    """
    main function to use the MLSMOTE
    """
    # Select columns from 'embedding_1' to 'embedding_768' for X
    X, y = df.loc[:,
        [f'embedding_{i + 1}' for i in range(768)]],df[['ARTHM','LE','None','RENT','TimeO']] # Creating a Dataframe
    X_sub, y_sub = get_minority_instace(X, y)  # Getting minority instance of that datframe
    X_res, y_res = MLSMOTE(X_sub, y_sub, 1000)  # Applying MLSMOTE to augment the dataframe
    print(df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    y_res = y_res[~((y_res['ARTHM'] == 1) | (y_res['None'] == 1))]
    # 将新生成的数据与原始数据拼接
    new_X = pd.concat([X, X_res], ignore_index=True)
    new_y = pd.concat([y, y_res], ignore_index=True)
    # 将拼接后的数据重新组合成一个 DataFrame
    result_df = pd.concat([new_X, new_y], axis=1)
    print(result_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    result_df = result_df.drop_duplicates()
    print(result_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    new_df = undersampling(result_df)
    print(new_df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].sum())
    new_df.to_csv('3_Data Encoding/train_split_final.csv')


