from math import log
from collections import defaultdict


def test(naive_bayes_classifier, test_data):
    counts = 0
    for features, gender in test_data:
        if gender == predict(naive_bayes_classifier, features):
            counts += 1
    return "{:.0%}".format(counts / len(test_data))


def train(features_and_gender):
    classes, frequency = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for features, gender in features_and_gender:
        classes[gender] += 1  # count the number of names in each class (800)
        for feature in features:
            frequency[gender, feature] += 1  # count the number of each feature in each class
    '''P(d_i |C) - probability to meet name with specific feature in each class'''
    for gender, feature in frequency:
        frequency[gender, feature] /= classes[gender]  # count the frequency for each feature
    '''P(C) - probability to meet one of two genders (0.5)'''
    for gender in classes:
        classes[gender] /= len(features_and_gender)
    return classes, frequency


def predict(naive_bayes_classifier, features):
    classes_and_frequency, features_and_frequency = naive_bayes_classifier
    prediction = max(classes_and_frequency.keys(), key=lambda gender: log(classes_and_frequency[gender]) + sum(
        log(features_and_frequency.get((gender, feature), 10 ** (-8))) for feature in features))
    return prediction


def sum_of_consonants(name):
    consonants = list("bcdfghjklmnpqrstvexz")
    count = 0
    for letter in name:
        if letter in consonants:
            count += 1
    return count


def sum_of_vowels(name):
    vowels = list("aeiouy")
    count = 0
    for letter in name:
        if letter in vowels:
            count += 1
    return count


def get_name_features(name):
    features = (
        'last_letter: %s' % name[-1],
        'length: %s' % len(name),
        'sum_of_consonants: %s' % sum_of_consonants(name),
        'sum_of_vowels: %s' % sum_of_vowels(name))
    return features


data = (line.split() for line in open(r'C:\Users\Iryna.Dosiak\PycharmProjects\AI\names.txt', encoding='utf8'))
features = [(get_name_features(features_data), gender) for features_data, gender in data]
train_data, test_data = features[:1600], features[-400:]
naive_bayes_classifier = train(train_data)
while True:
    print('\n_______________Menu_______________')
    print("1 - Predict gender by name")
    print("2 - Show accuracy by testing")
    print("3 - Exit")
    print('__________________________________')
    choice = int(input("Your choice: "))
    if choice == 1:
        name = input("Who are you: ")
        features = get_name_features(name)
        print(name + ', I think you are a nice ' + predict(naive_bayes_classifier, features) + '!')
    elif choice == 2:
        print('Accuracy: ', test(naive_bayes_classifier, test_data))
    else:
        break
