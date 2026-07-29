<H3>ENTER YOUR NAME: Junaid Sardar S</H3>
<H3>ENTER YOUR REGISTER NO.212224100028 </H3>
<H3>EX. NO.1</H3>
<H3>DATE:27/07/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
# ==========================================================
# EXPERIMENT NO.1
# Introduction to Kaggle and Data Preprocessing
# ==========================================================

# STEP 1 : Import Libraries

import os
import pandas as pd
import numpy as np
import kagglehub

from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================================
# STEP 2 : Import Dataset
# ==========================================================

path = kagglehub.dataset_download("brendan45774/test-file")

csv_file = os.path.join(path, "tested.csv")

df = pd.read_csv(csv_file)

# ==========================================================
# DATASET
# ==========================================================

print("=" * 80)
print("DATASET")
print("=" * 80)

display(df)

print("\nDataset Shape :", df.shape)

# ==========================================================
# NULL VALUES BEFORE PREPROCESSING
# ==========================================================

print("\n")
print("=" * 80)
print("NULL VALUES")
print("=" * 80)

null_values = pd.DataFrame({
    "Column Name": df.columns,
    "Missing Values": df.isnull().sum().values
})

display(null_values)

# ==========================================================
# STEP 3 : Handle Missing Values
# ==========================================================

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df.drop("Cabin", axis=1, inplace=True)

print("\n")
print("=" * 80)
print("NULL VALUES AFTER PREPROCESSING")
print("=" * 80)

null_after = pd.DataFrame({
    "Column Name": df.columns,
    "Missing Values": df.isnull().sum().values
})

display(null_after)

# ==========================================================
# STEP 4 : Encode Categorical Data
# ==========================================================

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

df["Embarked"] = encoder.fit_transform(df["Embarked"])

# ==========================================================
# Remove Unnecessary Columns
# ==========================================================

df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# ==========================================================
# STEP 5 : Normalize Data
# ==========================================================

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\n")
print("=" * 80)
print("NORMALIZED DATA")
print("=" * 80)

display(df)

# ==========================================================
# STEP 6 : Split Dataset
# ==========================================================

X = df.drop("Survived", axis=1)

y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n")
print("=" * 80)
print("DATA SPLITTING")
print("=" * 80)

split_table = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Feature Shape": [X_train.shape, X_test.shape],
    "Label Shape": [y_train.shape, y_test.shape]
})

display(split_table)

# ==========================================================
# TRAIN DATA
# ==========================================================

print("\n")
print("=" * 80)
print("TRAIN DATA")
print("=" * 80)

display(X_train)

print("\nTraining Labels")

display(y_train.to_frame(name="Survived"))

# ==========================================================
# TEST DATA
# ==========================================================

print("\n")
print("=" * 80)
print("TEST DATA")
print("=" * 80)

display(X_test)

print("\nTesting Labels")

display(y_test.to_frame(name="Survived"))

# ==========================================================
# FINAL DATASET
# ==========================================================

print("\n")
print("=" * 80)
print("FINAL PREPROCESSED DATASET")
print("=" * 80)

display(df)

print("\nExperiment Completed Successfully.")
```


## OUTPUT:
## DATASET
![alt text](image.png)

## NULL VALUES
![alt text](image-1.png)

## NULL VALUES AFTER PREPROCESSING
![alt text](image-2.png)

## NORMALIZED DATA
![alt text](image-3.png)

## TRAIN DATA
![alt text](image-4.png)

## TEST DATA
![alt text](image-5.png)

## FINAL PREPROCESSED DATASET
![alt text](image-6.png)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


