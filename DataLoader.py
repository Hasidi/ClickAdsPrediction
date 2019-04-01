import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(data_file_path, n_to_read=None):
    data_df = pd.read_csv(data_file_path, nrows=n_to_read)
    return data_df


def split_data(df, class_col_name):
    """
    split to train and test set with Stratified Shuffle
    :param df:
    :param class_col_name: target classification variable
    :return: x_train_Set, x_test_set, y_train_set, y_test_set
    """
    y = df.loc[:, class_col_name]
    X = df.drop(class_col_name, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test


def under_sampling_majority(x_train, y_train, class_col_name):
    """
    under sampling the majority class value samples
    :param x_train:
    :param y_train:
    :param class_col_name:
    :return:
    """
    X = pd.concat([x_train, y_train], axis=1)
    not_clicked = X[X[class_col_name] == 0]
    clicked = X[X[class_col_name] == 1]
    not_clicked_down_sampled = resample(not_clicked,
                                       replace=False,  # sample without replacement
                                       n_samples=len(clicked),  # match minority n
                                       random_state=27)  # reproducible results
    # combine minority and down_sampled majority
    down_sampled = pd.concat([not_clicked_down_sampled, clicked])
    # checking counts
    # down_sampled['click'].value_counts()
    train_y = down_sampled[class_col_name]
    train_X = down_sampled.loc[:, ~down_sampled.columns.isin([class_col_name])]
    assert len(train_y) == train_X.shape[0]
    return train_X, train_y








