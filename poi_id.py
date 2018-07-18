#!/usr/bin/python

import sys
import pickle
import numpy as np
from dataset_info import DatasetInfo
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


def convert_to_int(value):
    if (isinstance(value, basestring) or np.isnan(value)):
        return 0
    return int(value)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    'poi',
    'salary',
    'deferred_income', 
    'deferral_payments',
    'loan_advances', 
    'bonus', 
    'long_term_incentive', 
    'director_fees',
    'expenses', 
    'total_payments',
    'exercised_stock_options',
    'restricted_stock',
    'restricted_stock_deferred',
    'total_stock_value',
    'from_messages',
    'to_messages',
    'from_this_person_to_poi',
    'from_poi_to_this_person',
    'shared_receipt_with_poi',
    'other'
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

is_poi = 0
is_not_poi = 0
for person in data_dict:
    if (data_dict[person]['poi']):
        is_poi += 1
    else:
        is_not_poi += 1

print("poi => " + str(is_poi))
print("non-poi => " + str(is_not_poi))

# print dataset info :)
di = DatasetInfo()
di.setup(data_dict)
di.print_dataframe_info()

### Task 2: Remove outliers
# di.plot_outlier()

# remove outlier
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for person in my_dataset:
    my_dataset[person]['total_messages'] = convert_to_int(my_dataset[person]['to_messages']) + convert_to_int(my_dataset[person]['from_messages'])
    my_dataset[person]['total_messages_with_poi'] = convert_to_int(my_dataset[person]['from_this_person_to_poi']) + convert_to_int(my_dataset[person]['from_poi_to_this_person'])
    if (my_dataset[person]['total_messages'] > 0):
        my_dataset[person]['poi_total_messages_ratio'] = my_dataset[person]['total_messages_with_poi'] / my_dataset[person]['total_messages']
    else:
        my_dataset[person]['poi_total_messages_ratio'] = 0

# selected_features_list = [
#     'poi',
#     'poi_total_messages_ratio',
#     'salary',
#     # 'total_payments',
#     # 'total_stock_value',
#     # 'exercised_stock_options',
#     # 'restricted_stock',
#     # 'expenses',
# ]

# selected_features_list = [
#     'poi',
#     # 'poi_total_messages_ratio',
#     'salary',
#     'total_payments',
#     # 'total_stock_value',
#     # 'exercised_stock_options',
#     # 'restricted_stock',
#     'expenses',
# ]

# selected_features_list = [
#     'poi',
#     'salary',
#     'deferred_income', 
#     'deferral_payments',
#     'loan_advances', 
#     'bonus', 
#     'long_term_incentive', 
#     'director_fees',
#     'expenses', 
#     'total_payments',
#     'exercised_stock_options',
#     'restricted_stock',
#     'restricted_stock_deferred',
#     'total_stock_value',
#     'from_messages',
#     'to_messages',
#     'from_this_person_to_poi',
#     'from_poi_to_this_person',
#     'shared_receipt_with_poi',
#     'other'
# ]

selected_features_list = [
    'poi',
    'poi_total_messages_ratio',
    'salary',
    'total_payments',
    'total_stock_value',
    'exercised_stock_options',
    'restricted_stock',
    'expenses',
]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# test_classifier(clf, my_dataset, selected_features_list)

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# test_classifier(clf, my_dataset, selected_features_list)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(criterion='entropy')
# test_classifier(clf, my_dataset, selected_features_list)

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# test_classifier(clf, my_dataset, selected_features_list)

from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3, p=3)
# test_classifier(clf, my_dataset, selected_features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

# finding the neighbors value
# clf = KNeighborsClassifier(n_neighbors=2)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=4)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=5)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=6)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=7)
# test_classifier(clf, my_dataset, selected_features_list)

# finding the weights value
# clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
# test_classifier(clf, my_dataset, selected_features_list)

# algorithm
# clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
# test_classifier(clf, my_dataset, selected_features_list)

# p value
# clf = KNeighborsClassifier(n_neighbors=3, p=1)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, p=2)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, p=3)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, p=4)
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, p=5)
# test_classifier(clf, my_dataset, selected_features_list)

# p + weight
# clf = KNeighborsClassifier(n_neighbors=3, p=3, weights='distance')
# test_classifier(clf, my_dataset, selected_features_list)
# clf = KNeighborsClassifier(n_neighbors=3, p=4, weights='distance')
# test_classifier(clf, my_dataset, selected_features_list)


# the final classifier
clf = KNeighborsClassifier(n_neighbors=3, p=3)
test_classifier(clf, my_dataset, selected_features_list)
# test_classifier(clf, my_dataset, selected_features_list, folds=80)
# test_classifier(clf, my_dataset, selected_features_list, folds=450)
# test_classifier(clf, my_dataset, selected_features_list, folds=666)
# test_classifier(clf, my_dataset, selected_features_list, folds=1500)
# test_classifier(clf, my_dataset, selected_features_list, folds=22000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, selected_features_list)



print('extra thingy for the feature scaling')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
clf = KNeighborsClassifier(n_neighbors=2, p=2, weights='distance')
clf = Pipeline([('scale', MinMaxScaler()), ('classifier', clf)])
test_classifier(clf, my_dataset, features_list)