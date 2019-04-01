
def predict_samples(classifier_name, trained_classifier, test_x):
    print(f'predict test samples with classifier: [{classifier_name}]...')
    res = trained_classifier.predict(test_x)
    return res


def train_model(classifier_name, classifier, train_x, train_y):
    print(f'train classifier: [{classifier_name}]...')
    trained_classifier = classifier.fit(train_x, train_y)
    return trained_classifier
