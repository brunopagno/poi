# Identify Fraud From Enron Email

### Notes to the reviewer
* I'm using windows and powershell to execute python code. So any `PS C:\` refers to that.

## (Q1) The overview
This is the final part of a Udacity's course on Data Science where we analyse, parse, and figure out a dataset in order to identify a POI (Person of Interest).

The dataset is a list of emails exposed after the largest corporate fraud incident in American History. Enron Corporation was one of the largest companies in the country but went bankrupt due to corruption and fraud. In this incident a reasonable amount of information was leaked to the public. The emails are a part of this leaked information and contain messages that we will explore in order to find the "People of Interest".

This project is guided by [this document](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub) provided by Udacity.

## (Q1) Data Exploration and Outliers

The data structure as shown by `print_dataframe_info`

```
PS C:\Users\bruno_pagno\udacity\ud120-projects\final_project> python .\poi_id.py
C:\Python27\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
<class 'pandas.core.frame.DataFrame'>
Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
Data columns (total 21 columns):
salary                       146 non-null object
to_messages                  146 non-null object
deferral_payments            146 non-null object
total_payments               146 non-null object
exercised_stock_options      146 non-null object
bonus                        146 non-null object
restricted_stock             146 non-null object
shared_receipt_with_poi      146 non-null object
restricted_stock_deferred    146 non-null object
total_stock_value            146 non-null object
expenses                     146 non-null object
loan_advances                146 non-null object
from_messages                146 non-null object
other                        146 non-null object
from_this_person_to_poi      146 non-null object
poi                          146 non-null bool
director_fees                146 non-null object
deferred_income              146 non-null object
long_term_incentive          146 non-null object
email_address                146 non-null object
from_poi_to_this_person      146 non-null object
dtypes: bool(1), object(20)
memory usage: 24.1+ KB
                          count unique    top freq
salary                      146     95    NaN   51
to_messages                 146     87    NaN   60
deferral_payments           146     40    NaN  107
total_payments              146    126    NaN   21
exercised_stock_options     146    102    NaN   44
bonus                       146     42    NaN   64
restricted_stock            146     98    NaN   36
shared_receipt_with_poi     146     84    NaN   60
restricted_stock_deferred   146     19    NaN  128
total_stock_value           146    125    NaN   20
expenses                    146     95    NaN   51
loan_advances               146      5    NaN  142
from_messages               146     65    NaN   60
other                       146     93    NaN   53
from_this_person_to_poi     146     42    NaN   60
poi                         146      2  False  128
director_fees               146     18    NaN  129
deferred_income             146     45    NaN   97
long_term_incentive         146     53    NaN   80
email_address               146    112    NaN   35
from_poi_to_this_person     146     58    NaN   60
```

This allows us to highlight a few details:
* The dataset has 146 people.
* Each of them has a set of information like bonuses, salary, etc
* Much of the data is "NaN", meaning much of the data is missing
* There are 18 poi and 128 non-poi
* Features like loan_advances, director_fees, restricted_stock_deferred are missing values for almost the entire dataset, making them good candidates for disconsideration.

## (Q1) The outlier

![Salary x Bonus plot](https://github.com/brunopagno/poi/blob/master/plot_salary_bonus.PNG?raw=true)

I noticed a user called "TOTAL". It is clearly an information that can mislead analysis, since it is aggregating others. The image above shows clearly how much off the values are for Salary x Bonus. It's only fair that we remove the outlier when trying to find POI.

## (Q2) Features

I tweaked a bit with the original list of features and created a few new ones:

```
total_messages = to_messages + from_messages
total_messages_with_poi = from_this_person_to_poi + from_poi_to_this_person
poi_total_messages_ratio = total_messages_with_poi / total_messages

selected_features_list = [
    'poi',
    'poi_total_messages_ratio',
    'total_payments',
    'total_stock_value',
    'exercised_stock_options',
    'restricted_stock',
    'salary',
    'expenses'
]
```

Still I feel there were no significant changes in the result with the created features. The results with and without the 'poi_total_messages_ratio' are in [results_first_try](https://github.com/brunopagno/poi/blob/master/results_first_try.md). But considering the feature adding did not change much I preferred to focus on tweaking the algorithms for better results.

Also, I picked the features with most unique values.

## (Q3) The Algorithm

Testing which algorithm to use is all about chosing the values for parameters in the different algorithms provided by sklearn lib. As mentioned in the section above, my first tests had a few results which allowed me to take a good first look on which algorithms seem to be better for this dataset [results_first_try](https://github.com/brunopagno/poi/blob/master/results_first_try.md).

| Algorithm | Accuracy | Precision | Recall | F1 | F2 | Total predictions | True positives | False positives | False negatives | True negatives |
| --------- | -------- | --------- | ------ | -- | -- | ----------------- | -------------- | --------------- | --------------- | -------------- |
| GaussianNB | 0.85180 | 0.40591 | 0.24050 | 0.30204 | 0.26184 | 5000 | 481 | 704 | 519 | 2296 |
| DecisionTreeClassifier | 0.81273 | 0.29622 | 0.29400 | 0.29511 | 0.29444 | 15000 | 588 | 1397 | 1412 | 11603 |
| RandomForestClassifier |0.85933 | 0.39964 | 0.10950 | 0.17190 | 0.12810 | 15000 | 219 | 329 | 1781 | 12671 |
| LogisticRegression | 0.68300 | 0.03289 | 0.04850 | 0.03920 | 0.04430 | 15000 | 97 | 2852 | 1903 | 10148 |
| KNeighborsClassifier | 0.88907 | 0.83871 | 0.20800 | 0.33333 | 0.24482 | 15000 | 416 | 80 | 1584 | 12920 |

At first look GaussianNB, RandomForest and KNeighbors have similar accuracy, but in precision KNeighbors wins by a large margin. I decided to pursue this algorithm and try to tweak it's parameters, since the precision is high I believe it is possible to make some kind of trade to raise recall and have a decent result.

## (Q4) Algorithm Tuning

Algorithm tuning is all about gaining those precious extra points by tweaking the algorithm's parameters. I change parameters and try to verify where I seem to be getting better results, and evaluate which parameters should be the best choice for this dataset.

Tweaking parameters:

### NEIGHBORS

```
KNeighborsClassifier(n_neighbors=2)
        Accuracy: 0.88213       Precision: 0.75439      Recall: 0.17200 F1: 0.28013     F2: 0.20341
        Total predictions: 15000        True positives:  344    False positives:  112   False negatives: 1656   True negatives: 12888

KNeighborsClassifier(n_neighbors=3)
        Accuracy: 0.88727       Precision: 0.67497      Recall: 0.29800 F1: 0.41346     F2: 0.33547
        Total predictions: 15000        True positives:  596    False positives:  287   False negatives: 1404   True negatives: 12713

KNeighborsClassifier(n_neighbors=4)
        Accuracy: 0.88200       Precision: 0.80105      Recall: 0.15300 F1: 0.25693     F2: 0.18253
        Total predictions: 15000        True positives:  306    False positives:   76   False negatives: 1694   True negatives: 12924

KNeighborsClassifier(n_neighbors=5)
        Accuracy: 0.88907       Precision: 0.83871      Recall: 0.20800 F1: 0.33333     F2: 0.24482
        Total predictions: 15000        True positives:  416    False positives:   80   False negatives: 1584   True negatives: 12920

KNeighborsClassifier(n_neighbors=6)
        Accuracy: 0.86720       Precision: 1.00000      Recall: 0.00400 F1: 0.00797     F2: 0.00500
        Total predictions: 15000        True positives:    8    False positives:    0   False negatives: 1992   True negatives: 13000

KNeighborsClassifier(n_neighbors=7)
        Accuracy: 0.86853       Precision: 0.88889      Recall: 0.01600 F1: 0.03143     F2: 0.01991
        Total predictions: 15000        True positives:   32    False positives:    4   False negatives: 1968   True negatives: 12996
```

* More than 5 seem to lower recall too much.
* 2, 4 and 5 still have a low recall value which is not good.
* 3 seems the nearest to having a recall > 0.3 even though precision lowers a bit so let's focus in it.

### WEIGHTS

```
KNeighborsClassifier(n_neighbors=3, weights='distance')
        Accuracy: 0.85940       Precision: 0.45746      Recall: 0.29300 F1: 0.35721     F2: 0.31570
        Total predictions: 15000        True positives:  586    False positives:  695   False negatives: 1414   True negatives: 12305

KNeighborsClassifier(n_neighbors=5, weights='distance')
        Accuracy: 0.87613       Precision: 0.59889      Recall: 0.21500 F1: 0.31641     F2: 0.24662
        Total predictions: 15000        True positives:  430    False positives:  288   False negatives: 1570   True negatives: 12712
```

* Did not change much accuracy.
* Lost quite a bit of precision.
* Did not get recall over 0.3.

### P VALUE

```
KNeighborsClassifier(n_neighbors=3, p=1)
        Accuracy: 0.89440       Precision: 0.78032      Recall: 0.28950 F1: 0.42232     F2: 0.33116
        Total predictions: 15000        True positives:  579    False positives:  163   False negatives: 1421   True negatives: 12837

KNeighborsClassifier(n_neighbors=3, p=2)
        Accuracy: 0.88727       Precision: 0.67497      Recall: 0.29800 F1: 0.41346     F2: 0.33547
        Total predictions: 15000        True positives:  596    False positives:  287   False negatives: 1404   True negatives: 12713

KNeighborsClassifier(n_neighbors=3, p=3)
        Accuracy: 0.88967       Precision: 0.68101      Recall: 0.32450 F1: 0.43955     F2: 0.36245
        Total predictions: 15000        True positives:  649    False positives:  304   False negatives: 1351   True negatives: 12696

KNeighborsClassifier(n_neighbors=3, p=4)
        Accuracy: 0.88393       Precision: 0.62369      Recall: 0.32650 F1: 0.42862     F2: 0.36089
        Total predictions: 15000        True positives:  653    False positives:  394   False negatives: 1347   True negatives: 12606

KNeighborsClassifier(n_neighbors=3, p=5)
        Accuracy: 0.87880       Precision: 0.59440      Recall: 0.28650 F1: 0.38664     F2: 0.31961
        Total predictions: 15000        True positives:  573    False positives:  391   False negatives: 1427   True negatives: 12609
```

* We finally got some recall over 0.3 with good precision! P = 3 and P = 4

### P VALUE AND WEIGHTS

```
KNeighborsClassifier(n_neighbors=3, p=3, weights='distance')
        Accuracy: 0.84500       Precision: 0.38739      Recall: 0.27950 F1: 0.32472     F2: 0.29599
        Total predictions: 15000        True positives:  559    False positives:  884   False negatives: 1441   True negatives: 12116

KNeighborsClassifier(n_neighbors=3, p=4, weights='distance')
        Accuracy: 0.83953       Precision: 0.36638      Recall: 0.27900 F1: 0.31678     F2: 0.29297
        Total predictions: 15000        True positives:  558    False positives:  965   False negatives: 1442   True negatives: 12035
```

* Now precision really lowered and recall is lower than 3, so I guess it's better to keep 'weights' in the default value.

It seems the best case would be running the algorithm with `n_neighbors = 3` and `p = 3`.

## (Q5) Validation

Validation is the step where we check how good is the model developed for predictions. A classic validation mistake is not splitting data between training and testing, leading to an overfitting of the classifier.

For my cross validation I used the one that was already setup in `tester.py`. It uses sklearn's `StratifiedShuffleSplit` which is a "a merge of StratifiedKFold and ShuffleSplit". It felt a reasonable and safe way to validate my setup for it is sophisticated enough and is being used to validate our work in `tester.py`. Statified sampling aims to split the dataset so that the parts are somewhat similar. Because our dataset is small and has several more "non-POI" than "POI" keeping this structure is a better way to evaluate than using a random sampling.

 The results varied a bit according to the number of "folds" passed to the cross validator.

| Folds | Accuracy | Precision | Recall | F1 | F2 | Total predictions | True positives | False positives | True negatives | False negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 0.89000 | 0.69444 | 0.31250 | 0.43103 | 0.35112 | 1200 | 50 | 22 | 110 | 1018 |
| 450 | 0.89141 | 0.70516 | 0.31889 | 0.43917 | 0.35812 | 6750 | 287 | 120 | 613 | 5730 |
| 666 | 0.89089 | 0.69836 | 0.31982 | 0.43872 | 0.35871 | 9990 | 426 | 184 | 906 | 8474 |
| 1500 | 0.89044 | 0.69258 | 0.32067 | 0.43837 | 0.35925 | 22500 | 962 | 427 | 2038 | 19073 |
| 22000 | 0.89015 | 0.69552 | 0.31320 | 0.43191 | 0.35189 | 330000 | 13781 | 6033 | 30219 | 279967 |

The results for Accuracy, precision and recall seem good enough and well over the proposed lower bound (0.3).

## (Q6) Evaluation & performance

The evaluation is made with basis on metrics evaluated with the classifier and the data presented. Sklearn is gives us a few nice metrics like Accuracy, Precision and Recall. For this dataset I believe the Recall and Precision are the most importants as they evaluate the number of True Positives in relation with False positives and negatives.

```
Precision => True Positives / (True Positives + False Positives) # ratio of POIs that were correctly identified in the 'positives class'
Recall => True Positives / (True Positives + False Negatives) # ratio of POIs that were correctly identified between all the 'true positives'
```

The results presented in section 5 (Validation) gave enough confidence to evaluate this model as a good model since it has Precision and Recall over 0.3.