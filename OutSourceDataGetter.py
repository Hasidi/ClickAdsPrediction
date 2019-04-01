from time import sleep
import pandas as pd
import requests

from DataLoader import load_data


def write_created_new_features():
    """
    loads the original data and add to it 2 columns related to out-source data- google app category and USA timezones.
    write it to a new file
    :return:
    """
    df_android = load_data(r'Data\android_bids_us.csv')
    update_app_categories(df_android)
    update_time_zones(df_android)
    df_android.to_csv(r'Data/df_with_new_features.csv')
    # return df_android


def update_app_categories(df_android):
    dic_app_cat = {}
    df = load_data(r'Data\AppCat.csv')
    df_clean = df[['app_Name', 'app_Cat']]
    df_clean.apply(lambda x: put_app_cat_to_dict(x, dic_app_cat), axis=1)
    print('finish build app_cat dict')
    df_user_state = df_android[['app_id']]
    df_android['app_cat'] = df_user_state.apply(lambda x: get_app_cat(dic_app_cat, x), axis=1)


def put_app_cat_to_dict(row, dict_geo):
    app_name = row.loc['app_Name']
    app_cat = row.loc['app_Cat']
    if app_name not in dict_geo:
        dict_geo[app_name] = app_cat


def get_app_cat(app_cat_dict, row):
    state_code = row.loc['app_id']
    if state_code not in app_cat_dict:
        print('error')
    return app_cat_dict[state_code]


def update_time_zones(df_android):
    dic_geo = {}
    df = load_data(r'Data\US_States_Timezones_clean.csv')
    df_clean = df[['State_Code', 'TimeZone_Code']]
    df_clean.apply(lambda x: put_geo_location_to_dict(x, dic_geo), axis=1)
    print('finish build geo dict')
    df_user_state = df_android[['user_state']]
    df_android['geo_location'] = df_user_state.apply(lambda x: get_geo_location(dic_geo, x), axis=1)


def put_geo_location_to_dict(row, dict_geo):
    state_code = row.loc['State_Code']
    timezone_code = row.loc['TimeZone_Code']
    if state_code not in dict_geo:
        dict_geo[state_code] = timezone_code


def get_geo_location(geo_dict, row):
    state_code = row.loc['user_state']
    if state_code not in geo_dict:
        print('error')
    return geo_dict[state_code]


def get_app_type():
    dic = {}
    df = load_data(r'Data\android_bids_us.csv')
    df_app_ids = df[['app_id']]
    app_ids_set = df_app_ids.unique()
    for app_id in app_ids_set:
        if app_id not in dic:
            app_cat = call_google_store(app_id)
            dic[app_id] = app_cat
            sleep(1)
    dt_res = pd.DataFrame(dic.items())
    dt_res.columns = ['app_Name', 'app_Cat']
    dt_res.to_csv(r'Data\AppCat.csv', index=False)


def call_google_store(app_name):
    category_res = ''
    try:
        link = f'https://play.google.com/store/apps/details?id={app_name}'
        f = requests.get(link)
        page_text = f.text
        cat = page_text.split("<a itemprop=\"genre\"")[1]
        mid_cat = cat.strip().split(" ")[0]
        category_res = ''.join([c for c in mid_cat if c.isupper()])
    except Exception as e:
        print(f'caught exception for app: [{app_name}]: exception: [{e.__str__()}]')
    return category_res


# res = call_google_store('com.mobilityware.freecell')
# print('category is: ' + res)
# get_time_zones()
write_created_new_features()