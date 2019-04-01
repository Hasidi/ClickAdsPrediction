from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import DataEncoder
import DataLoader
import DataPreProcessor
import Evaluator
import ModelProcessor


eval_classifiers = {
    # 'RandomForestClassifier': RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=42),
    # 'TreeClassifier': DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=4, random_state=42),
    # 'LogisticRegression': LogisticRegression(penalty='l1', max_iter=100, random_state=42),
    'GBTrees': GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=1000, random_state=42,
                                          min_samples_split=2),
    # 'KNN': KNeighborsClassifier(n_neighbors=4, p=2)
}

trained_models = {}
models_scores = {}


def run():
    n = 100000
    df_android = DataLoader.load_data(r'Data\df_Ready_Data2.csv')
    print('finish read data.')
    # df_android = DataPreProcessor.drop_columns(df_android, ['user_isp_new'])
    df_android = df_android.dropna(how='any')
    col_label_encode = ['user_state', 'user_isp','app_cat','app_domain']
    df_android = DataEncoder.label_encoder(df_android, col_label_encode)
    col_one_hot_encode = ['device_maker', 'geo_location', 'day_of_week', 'part_of_day']
    # df_android = DataEncoder.encode_one_hot(df_android, list(df_android.columns[1:]))
    df_android = DataEncoder.encode_one_hot(df_android, col_one_hot_encode)
    print('finish encode data.')
    x_train, x_test, y_train, y_test = DataLoader.split_data(df_android, 'click')
    print('finish split to train and test')
    x_train, y_train = DataLoader.under_sampling_majority(x_train, y_train, 'click')
    print('finish sample data')

    print('start training classifiers...')
    for classifier_name, classifier in eval_classifiers.items():
        trained_model = ModelProcessor.train_model(classifier_name, classifier, x_train, y_train)
        trained_models[classifier_name] = trained_model
    print('Finish train classifiers.')
    print('start evaluating classifiers...')
    for model_name, model in trained_models.items():
        predictions = ModelProcessor.predict_samples(model_name, model, x_test)
        score = Evaluator.evaluate_performance_metric('auc', predictions, y_test)
        print(f'metric auc- score for [{model_name}]: [{score}]')


def write_ready_data():
    """
    load the original data with the extra out-source data and write new dataset after pre-processing stage.
    write the new data frame to a new file to be used by the learning and prediction method
    :return:
    """
    df_android = DataLoader.load_data(r'Data\df_with_new_features.csv')
    cleaned_data_df = DataPreProcessor.pre_process_data(df_android)
    cleaned_data_df.to_csv(r'Data/df_Ready_Data2.csv', index=False)


# write_ready_data()
run()