# Artificial Neural Networks (ANN) for Demographic Income Classification

This repository contains the Python implementation code for CSC3034 Computational Intelligence - Assignment 2. It is an implementation of an ANN using the `scikit-fuzzy` & `tensorflow` Python library.

The aim is to predict the outcome of whether an individual’s annual income will exceed $50,000 based on demographic and employment characteristics. The dataset used is the **Adult (Census Income)** dataset from the **UCI Machine Learning Repository** in this link: https://archive.ics.uci.edu/dataset/2/adult

Install these dependencies before running the code: `pip install numpy pandas matplotlib seaborn scikit-learn tensorflow ucimlrepo`

## Output Format

Below is how the output would look like when running the code

```
Loading dataset via ucimlrepo...

UCI Adult Dataset (ID 2) loaded

===============================
   BASIC DATASET INFORMATION
===============================

Dataset Shape: 48842 rows × 15 columns

Column Summary:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48842 entries, 0 to 48841
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   age             48842 non-null  int64
 1   workclass       47879 non-null  object
 2   fnlwgt          48842 non-null  int64
 3   education       48842 non-null  object
 4   education-num   48842 non-null  int64
 5   marital-status  48842 non-null  object
 6   occupation      47876 non-null  object
 7   relationship    48842 non-null  object
 8   race            48842 non-null  object
 9   sex             48842 non-null  object
 10  capital-gain    48842 non-null  int64
 11  capital-loss    48842 non-null  int64
 12  hours-per-week  48842 non-null  int64
 13  native-country  48568 non-null  object
 14  income          48842 non-null  object
dtypes: int64(6), object(9)
Memory Usage:
26.44817352294922 MB

First 5 rows:
   age         workclass  fnlwgt  education  education-num  ... capital-gain capital-loss hours-per-week native-country income
0   39         State-gov   77516  Bachelors             13  ...         2174            0             40  United-States  <=50K
1   50  Self-emp-not-inc   83311  Bachelors             13  ...            0            0             13  United-States  <=50K
2   38           Private  215646    HS-grad              9  ...            0            0             40  United-States  <=50K
3   53           Private  234721       11th              7  ...            0            0             40  United-States  <=50K
4   28           Private  338409  Bachelors             13  ...            0            0             40           Cuba  <=50K

[5 rows x 15 columns]

Numerical Statistics:
                  count       mean        std      min       25%       50%       75%        max
age             48842.0      38.64      13.71     17.0      28.0      37.0      48.0       90.0
fnlwgt          48842.0  189664.13  105604.03  12285.0  117550.5  178144.5  237642.0  1490400.0
education-num   48842.0      10.08       2.57      1.0       9.0      10.0      12.0       16.0
capital-gain    48842.0    1079.07    7452.02      0.0       0.0       0.0       0.0    99999.0
capital-loss    48842.0      87.50     403.00      0.0       0.0       0.0       0.0     4356.0
hours-per-week  48842.0      40.42      12.39      1.0      40.0      40.0      45.0       99.0

Categorical Statistics:
                count unique                 top   freq
workclass       47879      9             Private  33906
education       48842     16             HS-grad  15784
marital-status  48842      7  Married-civ-spouse  22379
occupation      47876     15      Prof-specialty   6172
relationship    48842      6             Husband  19716
race            48842      5               White  41762
sex             48842      2                Male  32650
native-country  48568     42       United-States  43832
income          48842      4               <=50K  24720

Duplicates:
Dataset contains 29 duplicate rows

=============================
   MISSING VALUES ANALYSIS
=============================

Missing values (NaN / Null):
workclass         963
occupation        966
native-country    274
dtype: int64

Rows with missing (NaN/Null) values: 1221

Unknown values (?):
workclass         1836
occupation        1843
native-country     583
dtype: int64

Rows with unknown (?) values: 2399

===========================================
   TARGET VARIABLE DISTRIBUTION (Income)
===========================================

Unique targets: ['<=50K' '>50K' '<=50K.' '>50K.']

Income Distribution:
income
<=50K    37155
>50K     11687
Name: count, dtype: int64

## Figure appears here (Income Distribution Piechart)

=====================================
   KEY FEATURES vs INCOME ANALYSIS
=====================================

## Figure appears here (Key Features vs Income Analysis Boxplots + Stacked Bars)

=================================
   NATIVE-COUNTRY DISTRIBUTION
=================================

Native-Country Distribution (Top 15):
native-country
United-States         43832
Mexico                  951
Philippines             295
Germany                 206
Puerto-Rico             184
Canada                  182
El-Salvador             155
India                   151
Cuba                    138
England                 127
China                   122
South                   115
Jamaica                 106
Italy                   105
Dominican-Republic      103
Name: count, dtype: int64

Total unique countries: 42
United-States count: 43832 (89.7%)
Other countries combined: 5010 (10.3%)

========================
   CORRELATION MATRIX
========================

## Figure appears here (Feature Correlation Matrix)

Cleaning dataset...

Stripped whitespaces
Removed 3620 rows with missing values
Removed 28 duplicate rows
Dropped 'fnlwgt' & 'education' columns
Standardized target variable (income)
Grouped non-US countries to a single category: 'Other'

Final dataset shape: 45194 rows × 13 columns

First 5 rows of cleaned dataset:
   age         workclass  education-num         marital-status  ... capital-loss hours-per-week native-country income
0   39         State-gov             13          Never-married  ...            0             40  United-States  <=50K
1   50  Self-emp-not-inc             13     Married-civ-spouse  ...            0             13  United-States  <=50K
2   38           Private              9               Divorced  ...            0             40  United-States  <=50K
3   53           Private              7     Married-civ-spouse  ...            0             40  United-States  <=50K
4   28           Private             13     Married-civ-spouse  ...            0             40          Other  <=50K

[5 rows x 13 columns]

Column Summary:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45194 entries, 0 to 45193
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   age             45194 non-null  int64
 1   workclass       45194 non-null  object
 2   education-num   45194 non-null  int64
 3   marital-status  45194 non-null  object
 4   occupation      45194 non-null  object
 5   relationship    45194 non-null  object
 6   race            45194 non-null  object
 7   sex             45194 non-null  object
 8   capital-gain    45194 non-null  int64
 9   capital-loss    45194 non-null  int64
 10  hours-per-week  45194 non-null  int64
 11  native-country  45194 non-null  object
 12  income          45194 non-null  object
dtypes: int64(5), object(8)
Memory Usage:
21.71961498260498 MB

Example categorical distributions:

Value counts for 'workclass':
workclass
Private             33281
Self-emp-not-inc     3795
Local-gov            3100
State-gov            1946
Self-emp-inc         1645
Federal-gov          1406
Without-pay            21
Name: count, dtype: int64

Value counts for 'marital-status':
marital-status
Married-civ-spouse       21048
Never-married            14580
Divorced                  6294
Separated                 1411
Widowed                   1277
Married-spouse-absent      552
Married-AF-spouse           32
Name: count, dtype: int64

Value counts for 'occupation':
occupation
Craft-repair         6015
Prof-specialty       6003
Exec-managerial      5982
Adm-clerical         5537
Sales                5408
Other-service        4805
Machine-op-inspct    2967
Transport-moving     2316
Handlers-cleaners    2045
Farming-fishing      1477
Tech-support         1419
Tech-support         1419
Protective-serv       976
Priv-house-serv       230
Armed-Forces           14
Name: count, dtype: int64

Value counts for 'income':
income
<=50K    33988
>50K     11206
Name: count, dtype: int64

Dataset cleaned

Preprocessing dataset...

Encoding target (income) as 0/1...
  Target Mapping: {'<=50K': np.int64(0), '>50K': np.int64(1)}

One-Hot Encoding categorical features...
   Features before encoding: 12
   Features after encoding: 41

Splitting data...
  Dataset split in 70:15:15 ratio

Scaling numerical columns: ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

Dataset preprocessed & split successfully

Constructing ANN Model...

Architecture Design:
   Input Features: 41
   Hidden Layers: 4 (256 → 128 → 64 → 32)
   Activation: ReLU
   Regularization: Dropout (0.3, 0.3, 0.2, 0.2)
   Output: 1 neuron (Sigmoid for binary classification)

Model architecture:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 256)                 │          10,752 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 54,017 (211.00 KB)
 Trainable params: 54,017 (211.00 KB)
 Non-trainable params: 0 (0.00 B)

Training Configuration:
   Optimizer: Adam (learning rate=0.001)
   Loss: Binary Crossentropy
   Metrics: Accuracy, Area Under Curve

Model constructed

Training model...
Epoch 1/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - AUC: 0.8903 - accuracy: 0.8403 - loss: 0.3486 - val_AUC: 0.9103 - val_accuracy: 0.8484 - val_loss: 0.3205 - learning_rate: 0.0010
Epoch 2/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - AUC: 0.9039 - accuracy: 0.8496 - loss: 0.3287 - val_AUC: 0.9117 - val_accuracy: 0.8487 - val_loss: 0.3203 - learning_rate: 0.0010
Epoch 3/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - AUC: 0.9059 - accuracy: 0.8523 - loss: 0.3252 - val_AUC: 0.9120 - val_accuracy: 0.8501 - val_loss: 0.3171 - learning_rate: 0.0010
Epoch 4/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - AUC: 0.9074 - accuracy: 0.8529 - loss: 0.3227 - val_AUC: 0.9108 - val_accuracy: 0.8500 - val_loss: 0.3163 - learning_rate: 0.0010
...
## more epochs shown here as training continues
...
Epoch 25/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - AUC: 0.9266 - accuracy: 0.8689 - loss: 0.2857 - val_AUC: 0.9103 - val_accuracy: 0.8517 - val_loss: 0.3177 - learning_rate: 2.5000e-04
Epoch 26/50
989/989 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - AUC: 0.9276 - accuracy: 0.8694 - loss: 0.2848 - val_AUC: 0.9106 - val_accuracy: 0.8516 - val_loss: 0.3158 - learning_rate: 2.5000e-04

Model trained

Generating predictions...
212/212 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  

=======================
   EVALUATION REPORT
=======================
Test Loss:     0.3150
Test Accuracy: 0.8565
Test AUC:      0.9112

Classification Report:
              precision    recall  f1-score   support
       <=50K       0.89      0.92      0.91      5099
        >50K       0.74      0.65      0.69      1681

    accuracy                           0.86      6780
   macro avg       0.81      0.79      0.80      6780
weighted avg       0.85      0.86      0.85      6780

## Figure appears here (Loss & Accuracy vs Epochs, Confusion Matrix & ROC)
```

Exact figures like accuracy, loss and number of epochs _WILL VARY_ as the model is rerun at different times. This is just the general output as an example.
