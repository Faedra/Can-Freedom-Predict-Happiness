## Can Freedom Predict Happiness? | A Machine Learning Exploration

### Project Background and Summary

Using Data fron the Human Freedom Index and the Happiness Index, I attempted to answer the question whether freedom measures correlate with happiness measures in a way that can be predicted, or at least provide actionable information. An exploration of machine learning techniques, building upon my earlier analysis comparing/contrasting these same datasets, to also answer whether these techniques will return similar or different insights from the original analysis. 

### Project Steps:

      * Data Prep
      * Feature Selection
      * Correlation + Residual Plots
   * Regression + Scoring Section 1: SKLearn and Train/Test/Split
   * Regression + Scoring Section 2: Keras and TensorFlow
   * Plotting value changes using matplotlib: https://keras.io/visualization/
   * Clustering and Forecasting in Tableau: https://tabsoft.co/2Zopdfd
   
### Data Sources:

#### HAPPINESS DATASETS:
World Happiness Report (2019) https://worldhappiness.report/ed/2019/

World Happiness Report data on Kaggle Includes happiness scores and rankings from the Gallup World Poll https://www.kaggle.com/unsdsn/world-happiness

#### TWO MERGED DATASETS on HUMAN FREEDOM:
Human Freedom Index from the Cato Institute Includes the basic freedom scores and rankings https://www.cato.org/human-freedom-index-new

Human Freedom Index exploration on Kaggle Breaks Cato data into smaller subsets for more detailed study: https://www.kaggle.com/gsutters/the-human-freedom-index#hfi_cc_2018.csv

### Libraries Used:

#### Basic Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### For feature selection/univariate selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

#### For feature selection/feature importance
from sklearn.ensemble import ExtraTreesClassifier

#### For feature selection/correlation matrix
import seaborn as sns

#### For feature importance using XG Boost
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance

#### For linear regression model + scoring with SKLearn: 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#### For regression with Keras and TensorFlow:
from keras import backend as K
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

### Findings

Display the same output as a dataframe:¶
Interestingly, this feature select method shows that the majority of relevant features relate to economic freedoms (7/10) rather than personal freedoms (3/10), which is the same conclusion we reached in Project 1 in the original exploration of this data.


1. **Get your data set**

   ![3-Data](Images/3-Data.png)

   The USGS provides earthquake data in a number of different formats, updated every 5 minutes. Visit the [USGS GeoJSON Feed](http://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) page and pick a data set to visualize. When you click on a data set, for example 'All Earthquakes from the Past 7 Days', you will be given a JSON representation of that data. You will be using the URL of this JSON to pull in the data for our visualization.

   ![4-JSON](Images/4-JSON.png)

2. **Import & Visualize the Data**

   Create a map using Leaflet that plots all of the earthquakes from your data set based on their longitude and latitude.

   * Your data markers should reflect the magnitude of the earthquake in their size and color. Earthquakes with higher magnitudes should appear larger and darker in color.

   * Include popups that provide additional information about the earthquake when a marker is clicked.

   * Create a legend that will provide context for your map data.

   * Your visualization should look something like the map above.

- - -

### Level 2: More Data (Optional)

![5-Advanced](Images/5-Advanced.png)

The USGS wants you to plot a second data set on your map to illustrate the relationship between tectonic plates and seismic activity. You will need to pull in a second data set and visualize it along side your original set of data. Data on tectonic plates can be found at <https://github.com/fraxen/tectonicplates>.

In this step we are going to..

* Plot a second data set on our map.

* Add a number of base maps to choose from as well as separate out our two different data sets into overlays that can be turned on and off independently.

* Add layer controls to our map.

- - -

### Assessment

Your final product will be assessed on the following metrics:

* Completion of assigned tasks

* Visual appearance

* Professionalism

**Good luck!**

## Copyright

Data Boot Camp (C) 2018. All Rights Reserved.








