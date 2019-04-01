from datetime import datetime
import pandas as pd

Morning = 6  # 0
Noon = 12  # 1
Evening = 18  # 2
Night = 0  # 3


# is_valid_version = lambda x: x['device_osv'].str.contains('^\d+(\.\d+)*$', regex=True)


def extract_features_from_existing(df):
    # handling time UTC
    data_names = df[['utc_time']].apply(lambda row: convert_utc_to_day(row), axis=1)
    print('finish convert utc_time')
    data_names = data_names.apply(pd.Series)
    data_names.columns = ['day_of_week', 'part_of_day']  # override default col names
    df = pd.concat([df[:], data_names.apply(pd.Series)[:]], axis=1)
    print('finish process utc_time')
    # handling os version
    df = df[df['device_osv'].str.contains('^\d+(\.\d+)*$', regex=True) == True]
    df['device_main_osv'] = df['device_osv'].apply(lambda x: x.split('.')[0])
    # cleaned_data_df['device_sub_osv'] = cleaned_data_df['device_osv'].apply(lambda x: convert_version_to_sub_version(x))

    # handling domain_app
    df['app_domain'] = df['app_id'].apply(lambda x: extract_app_domain(x))

    # app_cat_dic = get_app_categories_dic()
    # df['app_cat'] = df['app_cat'].apply(lambda x: extract_app_category(app_cat_dic, x))
    # df = reduce_categorial_features_values(df)
    return df


def convert_utc_to_day(utc_dt):
    """
    convert a utc time to 2 values: day of the week and part of the day
    :param utc_dt:
    :return:
    """
    # date_str = datetime.fromtimestamp(utc_dt / 1000).strftime('%Y-%m-%d %H:%M:%S')
    # day_of_week = pd.to_datetime(pd.Series([date_str])).dt.day_name()
    date_obj = datetime.fromtimestamp(utc_dt / 1000)
    day_of_week = pd.to_datetime(date_obj).weekday_name
    hour_time = date_obj.hour
    if Morning <= hour_time < Noon:
        part_of_day = 'Morning'
    elif Noon <= hour_time < Evening:
        part_of_day = 'Noon'
    elif Evening <= hour_time < Night:
        part_of_day = 'Evening'
    else:
        part_of_day = 'Night'
    return day_of_week, part_of_day


def create_screen_ratio_feature(df):
    """
    creates new column of wide screen or not based on device width and height
    :param df:
    :return:
    """
    df['screen_ratio_type'] = df[['device_width', 'device_height']].apply(
        lambda x: calc_screen_ratio(x['device_width'], x['device_height']), axis=1)
    return df


def calc_screen_ratio(width, height):
    if width >= height:
        ratio = width / height
    else:
        ratio = height / width
    if ratio >= 1.7:
        res = 'wide'
    else:
        res = 'non_wide'
    return res


def get_app_categories_dic():
    communication = ['COMMUNICATION', 'NEWS']
    social = ['SOCIAL', 'DATING']
    entertainment = ['LIFESTYLE', 'TRAVEL', 'FOOD', 'HEALTH', 'PHOTO', 'ART', 'SHOPPING', 'HOUSE', 'WEATHER', 'MAP',
                     'BEAUTY', 'VIDEO', 'AUTOANDVEHICLES', 'MUSIC', 'COMICS', 'PERSONALIZATION', 'SPORTS']
    business = ['BUSINESS', 'FINANCE']
    tools = ['TOOLS', 'LIBRARIES']
    education = ['EDUCATION', 'BOOK']
    game = ['GAME']
    app_categories = {'communication': communication, 'social': social, 'entertainment': entertainment,
                      'business': business, 'tools': tools, 'education': education,
                      'game': game}
    return app_categories


def extract_app_category(categories_dic, val):
    """
    convert original google store category to related categories in a given list
    :param categories_dic:
    :param val:
    :return:
    """
    for k, values in categories_dic.items():
        for v in values:
            if val.startswith(v):
                return k
    return ''


def extract_app_domain(val):
    # domains = ['com', 'org', 'net']
    domains = ['com', 'org', 'net', 'us', 'int', 'ru', 'it', 'uk', 'eu', 'fr']
    splitted = val.split('.')
    for domain in domains:
        for split in splitted:
            if domain in split:
                return domain
    return ''


def reduce_categorial_features_values(df):
    """
    drops rows related to device maker and user_isp
    :param df:
    :return:
    """
    device_makers_counts = df['device_maker'].value_counts()
    device_makers = list(device_makers_counts[device_makers_counts > 1000].index)
    df['device_maker_new'] = df['device_maker'].apply(lambda x: x if x in device_makers else 'other')

    isp_counts = df['user_isp'].value_counts()
    isps = list(isp_counts[isp_counts > 10000].index)
    df['user_isp_new'] = df['user_isp'].apply(lambda x: x if x in isps else 'other')

    df['screen_ratio_type'] = df[['device_width', 'device_height']].apply(
        lambda x: calc_screen_ratio(x['device_width'], x['device_height']), axis=1)
    return df


def convert_version_to_sub_version(version: str) -> str:
    if '.' in version:
        return version.split('.')[1]
    return str(0)


