## Can Freedom Predict Happiness? | A Machine Learning Exploration

### Project Background and Summary

Using Data fron the Human Freedom Index and the Happiness Index, I attempted to answer the question: do freedom measures correlate with happiness measures in a way that can be predicted, or at least provide actionable information? An exploration of machine learning techniques, building upon my earlier analysis comparing/contrasting these same datasets, to also answer whether these techniques will return similar or different insights from the original analysis. 

### Project Steps:

   * Data Prep
   * Feature Selection
   * Correlation + Residual Plots
   * Regression + Scoring Section 1: SKLearn and Train/Test/Split
   * Regression + Scoring Section 2: Keras and TensorFlow
   * Plotting value changes using matplotlib: https://keras.io/visualization/
   * Clustering and Forecasting in Tableau: https://help.tableau.com/current/pro/desktop/en-us/clustering.htm
   
### Data Sources:

1. #### Happiness Datasets:
     * [World Happiness Report (2019)](https://worldhappiness.report/ed/2019/)
     * [Additional World Happiness Report data on Kaggle](https://www.kaggle.com/unsdsn/world-happiness) (includes happiness scores and rankings from the Gallup World Poll)

2. #### Human Freedom Datasets:
     * [Human Freedom Index from the Cato Institute](https://www.cato.org/human-freedom-index-new) (includes the basic freedom scores and rankings) 
     * [Human Freedom Index exploration on Kaggle](https://www.kaggle.com/gsutters/the-human-freedom-index#hfi_cc_2018.csv) (breaks Cato data into smaller subsets for more detailed study)

### Libraries Used:

1. #### Basic Libraries:
    ```
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ```
2. #### Feature selection (univariate selection and correlation matrix)
    ```
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_regression
    import seaborn as sns
    ```
3. #### Feature importance using XG Boost
    ```
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    ```
4. #### Linear regression model + scoring with SKLearn: 
    ```
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import  LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    ```
5. #### For regression with Keras and TensorFlow:
    ```
    from keras import backend as K
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.layers import Dense
    ```

### Findings

Interestingly, feature select methods show that the majority of top-10 relevant features relate to economic freedoms (7/10) rather than personal freedoms (3/10), which is the same conclusion we reached in Project 1 in the original exploration of this data. Additionally, 
