from sklearn.preprocessing import LabelEncoder
import pandas as pd


def encode_one_hot(df, col_names) -> pd.DataFrame:
    """each
    encode each categorical feature to one hot encoding. the original column is deleted
    :param df:
    :param col_names:
    :return:
    """
    for col_name in col_names:
        encoded = pd.get_dummies(df[col_name], prefix=col_name, drop_first=False)
        df = pd.concat([df, encoded], axis=1)
        df = df.drop([col_name], axis=1)
    return df


def label_encoder(df, col_names: list) -> pd.DataFrame:
    """
    encode each categorical feature to int encoding. the original column is deleted
    :param df:
    :param col_names:
    :return:
    """
    le = LabelEncoder()
    for col_name in col_names:
        df.loc[:, col_name] = le.fit_transform(df[col_name])
    return df
