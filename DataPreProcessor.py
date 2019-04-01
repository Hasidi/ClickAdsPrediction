import FeatureEngineer
import pandas as pd
import numpy as np


def pre_process_data(df):
    # df = drop_columns(df, [df.columns[0], 'bidid', 'user_state', 'device_model', 'marketplace'])
    df = drop_columns(df, [df.columns[0], 'bidid', 'device_model', 'marketplace'])
    df = df.dropna(how='any')
    df = reduce_exist_features_values_rows(df)
    if df.shape[0] == 0:
        print('empty dataframe cant continue.......................')
        return None
    df = FeatureEngineer.extract_features_from_existing(df)
    # print('fill missing values..')
    # df = fill_missing_values(df)
    print('enter dropping columns')
    # df = drop_columns(df, ['app_id', 'utc_time', 'user_isp', 'device_maker',
    #                        'device_osv', 'device_height', 'device_width'])
    df = drop_columns(df, ['app_id', 'utc_time',
                           'device_osv'])
    # print('dropping unknown values categories..')
    # filter_unknown_categories = df['app_cat'] == 'unknown'
    # df = filter_data_by_condition(df, filter_unknown_categories)
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    res = df.drop(labels=columns, axis=1)
    return res


def fill_missing_values(df: pd.DataFrame):
    for column in df:
        missing_data = df[column].loc[df[column].isnull()]
        if len(missing_data) > 0:
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(np.mean(df[column]))
            else:
                print('col to: ' + column)
                common_value = df[column].mode().iloc[0]
                print(f'col [{column}] common value [{common_value}]')
                df[column] = df[column].fillna(common_value)
    return df


def filter_data_by_condition(df, condition_func) -> pd.DataFrame:
    # res = df.loc[lambda x: condition_func(x)]
    res = df.loc[~condition_func]
    return res


def reduce_exist_features_values_rows(df):
    device_makers_counts = df['device_maker'].value_counts()
    device_makers = list(device_makers_counts[device_makers_counts > 1000].index)
    df = df[df.device_maker.isin(device_makers)]
    # df['device_maker_new'] = df_tmp['device_maker']

    isp_counts = df['user_isp'].value_counts()
    isps = list(isp_counts[isp_counts > 10000].index)
    df = df[df.user_isp.isin(isps)]
    # df['user_isp_new'] = df_tmp['user_isp']
    return df

