
# Activity Recognition

## 1. Documentation

> This section is about the most important parts of the developed package called **actreg** and all the features given by the API.  
You get in touch with its main modules and functionalities.

### 1.1 SequentialSensoryDataModel
> The model called **SequentialSensoryDataModel** is designed for processing sensory data that was measured in a sequential setting (time dependand), classified by e.g. the activity of the user.   
In this use case the class was used in orde3r to recognize certain user activity based on sequentially measured vital signs.

#### 1.1.1. fit
At the beginning the model splits the input matrix and output vector in training and validation parts in order to compare the accurracies of different models. It tests the 
 predictive power of the well-known models (KNeighborsClassifier, GradientBoostingClassifier and SVC) during the fitting process. and sets the best performing one for further computations.  
 It runs all computations in parallel with usage of the **joblib** package in order to finish the fitting as fast as possible.
  
#### 1.1.2. predict
Prediction of the best performing model.

#### 1.1.3. score
Score of the best performing model

#### 1.1.4. normalize
For normalization it uses the **StandardScaler** of sklearn package. For applying nomralization with the model scaler the class provides a function to normalize the data before processing.

#### 1.1.5. sequentialize
With the sequentialize function the user can transform its input raw data (input matrix, output vectors) into a sequentialized vector matrix by using the models internal preprocessing object **Sequentializer** in order to have consistent sequence lengths. Custom sequence lengths could be passed during the model initialization (default 50).

#### 1.1.6. get_models
Returns all the models computed during the model selection.

#### 1.1.7. set_model
Sets another of the computed models as best performing one if the user wants to do so (knn, boosting or svc)

#### 1.1.8. save_model
For further usage the user of this class can save the model and all its components (best performing model, other models, standard scaler, sequentializer)

#### 1.1.9. load_model
In order to load a precomputed model one can use this function by just giving the path of the model and all its components get loaded.

### 1.2. ScoreMap

> For having a mechanism to classify the actions of a user and them give a certain score the class called **ScoreMap** was introduced.   
It needs a vector of categories and the ordinal scale depending on the cetagories as features as well as a scoring strategy (linear, quadratic, kubic, exponential) for initialization.

#### 1.2.1. fit
Fits categories to scores and enumerates them.

#### 1.2.2. get_activitiy
Returns the description of an activity as string for a certain number.

#### 1.2.3 get_score
Depending on the scoring strategy this function returns the score for a certain number (defined by enumeration of the categories)

## 2. Usage

> In this sector one can find some example of how to use the model and score map.

### 2.1. Model training
At first import all needed dependencies:
```python
import pandas as ps
import numpy as np
from sklearn.model_selection import train_test_split

# The new custom package
from actreg.seqmod import SequentialSensoryDataModel
```
For the model fitting simply follow this example:
````python
model = SequentialSensoryDataModel()

input_files = ['data_collection/labelled/mHealth_subject' + str(int(i)) + '.csv' for i in np.linspace(1, 10, 10)]

# Load the data from csv files
X, y = np.array([]), np.array([])
for file in input_files:
    XY = ps.read_csv(file, sep=',').as_matrix()
    X_tmp, y_tmp = model.sequentialize(XY[:, :-1], XY[:, -1])
    X = np.append(X, X_tmp, axis=0) if X.any() else X_tmp
    y = np.append(y, y_tmp) if y.any() else y_tmp

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y)

# Fit the model to the training dataset
model.fit(X_train, y_train)

# Test the model with best parameter and normalized test data
X_test = model.normalize(X_test)
print('\nTest score:', model.score(X_test, y_test))

# Safe the model and all his components
print('Save best model...\n')
model.save_model(path=save_path)
````

### 2.2. Model prediction

For making predictions with an precomputed model follow this example:
````python
file = 'data_collection/non_labelled/mHealth_non_labelled_subject1.csv'

# Load data
X = ps.read_csv(file, sep=',').as_matrix()
X = model.sequentialize(X)

# Normalize the data
X = model.normalize(X)

# Predict on the data
prediction = model.predict(X)
````

### 2.3. Score the predictions
For calculating wether the current user is acting quite active or not follow this example:
````python
# Compute weights (percentage of whole data) for the scores and plot results
count = len(prediction)
weights = {}

for activity in np.unique(prediction):
    weights[activity] = len(list(filter(lambda x: x == activity, prediction))) / count
    
# Calculate the total score
total_score = 0
for key in weights:
    score = score_map.get_score(key)
    total_score += score * weights[key]
print(total_score)
````