#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

from modules.data_preprocess import DataProcessor
from modules.regression import MovingTimeRegression, ElapsedTimeRegression
from modules.clustering import DifficultyLevelClustering


# In[2]:


dataRoute = './data/'
cleanedRoute = './data/cleaned/'
resultsRoute = './results/'
profiledRoute = './profiling/'
regressionModelRoute = './results/models/regression/'
clusteringModelRoute = './results/models/clustering/'
scalerRoute = './results/scalers/'
paramsRoute = './results/params/'
csvFileName = 'data_2019_03_20.csv'
cleanedFileName = 'cleaned_data_2019_03_20.csv'
describeFileName = 'data_2019_03_20_DESCRIPTION.csv'
cleanedDescribeName = 'advanced_2019_03_20_DESCRIPTION.csv'


# # Data processing

# In[ ]:


dataProc = DataProcessor(dataRoute, cleanedRoute, profiledRoute, resultsRoute, csvFileName, describeFileName, cleanedDescribeName)


# In[ ]:


dataProc.readRawData()


# In[ ]:


dataProc.dropUselesColumns()


# In[ ]:


dataProc.detectAndDropOutliers()


# In[ ]:





# In[ ]:


len(dataProc.rawData[dataProc.rawData['workout_type'] < 10])


# In[ ]:


len(dataProc.rawData[dataProc.rawData['age_group'] == 0])


# In[ ]:





# In[ ]:


dataProc.showWeeklyInsightGraph(language='hun')


# In[ ]:


dataProc.showDailyInsightGraph(language='hun')


# In[ ]:





# In[ ]:


dataProc.oneHotEncodeFeatures()


# In[ ]:


dataProc.saveAndProfileCleanedData()


# In[ ]:





# # Regressions

# ### Moving time

# In[3]:


movingTime = MovingTimeRegression(cleanedRoute, regressionModelRoute, scalerRoute, paramsRoute, cleanedFileName)


# In[4]:


movingTime.prepareData()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# WARNING : this function can take hours, and it does not have to be \n# executed because the generated files can be found under:\n# ./data/results/params\n\n# change the following variable to True to re-calculate the params\nRUN_LONG_MOVING_TIME_CALCULATION = False\nif RUN_LONG_MOVING_TIME_CALCULATION:\n    movingTime.calulateBestParamsForRandomForest()')


# In[ ]:





# In[5]:


get_ipython().run_cell_magic('time', '', 'movingTime.trainModels(verbose=True, writeToFile=True)')


# In[8]:


# the all prediction func. is excepting a list with 27 elements, where the values shoud
# represent the columns of the allTrainX or the dataset (in the second case ignore
# the moving_time and the elapsed_time columns)
# easy test case:
# - select a row from dataset with: mlRegression.dataset.iloc[i]
# - use: testData.drop(labels=['moving_time','elapsed_time','average_speed']).tolist() as param of the function
# a generated list: TODO
testData = movingTime.dataset.iloc[100]
print('predicted values')
movingTime.getPredictionWithAllModels(testData.drop(labels=['moving_time','elapsed_time','average_speed']).tolist())
print('original value:')
print(testData['moving_time'])


# In[ ]:


testData['moving_time']


# In[ ]:


# the base prediction function except a list with 5 element, where the values should
# represent: distance, elev_high, elev_low, total_elevation_gain, trainer_onehot
# example 1: [22000, 500, 100, 700, 1.0]
# example 2: [102977.0, 476.0, 241.0, 968.0, 0.0]
movingTime.getPredictionWithBaseModels([102977.0, 476.6, 241.6, 968.0, 0.0])


# In[ ]:


# to get predictions without retraining the models call 
# loadTrainedModelsAndScalers() and then one of the prediction making functions


# In[ ]:





# ### Elapsed time

# In[3]:


elapsedTime = ElapsedTimeRegression(cleanedRoute, regressionModelRoute, scalerRoute, paramsRoute, cleanedFileName)


# In[4]:


elapsedTime.prepareData()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# WARNING : this function will take several hours, and it does not have to be \n# executed because the generated files can be found under:\n# ./data/results/params\n\n# change the following variable to True to re-calculate the params\nRUN_LONG_ELAPSED_TIME_CALCULATION = False\nif RUN_LONG_ELAPSED_TIME_CALCULATION:\n    elapsedTime.calulateBestParamsForRandomForest()')


# In[ ]:





# In[5]:


elapsedTime.trainModels(verbose=True, writeToFile=False)


# In[7]:


len(elapsedTime.userTestX)


# In[ ]:


elapsedTime.loadTrainedModelsAndScalers()


# In[ ]:


len(elapsedTime.userTrainX.columns)


# In[ ]:


# to get predictions without retraining the models call 
# loadTrainedModelsAndScalers() and then one of the prediction making functions


# In[ ]:





# In[ ]:





# # Clustering

# In[3]:


difficultyLevels = DifficultyLevelClustering(cleanedRoute, clusteringModelRoute, resultsRoute, cleanedFileName)


# In[4]:


difficultyLevels.prepareData()


# In[ ]:





# ### Age group 0

# In[6]:


difficultyLevels.trainAgeNullModels(writeToFile=False, graphLang='hu')


# In[ ]:





# ### Age group 1

# In[7]:


difficultyLevels.trainAgeOneModels(writeToFile=False, graphLang='hu')


# In[ ]:





# ### Age group 2

# In[9]:


difficultyLevels.trainAgeTwoModels(writeToFile=True, graphLang='hu')


# In[ ]:





# In[ ]:





# In[ ]:


# after training the models for the three age groups 
# OR 
# loading the saved models with: TODO function call
# call getTrainingSuggestions to get a suggestion for training.
# params:( 
#   ageGroup        - 0,1,2
#   clustering      - kmeans, minibatch, spectral
#   trainType       - A, B, C, D, E
#   trainDifficulty - 0.25, 0.5, 0.75
# )


# In[79]:


suggestion = difficultyLevels.getTrainingSuggestions(2, 'kmeans', 'D', 0.25)


# In[80]:


difficultyLevels.mps_to_kmph(suggestion['average_speed'])


# In[ ]:





# In[ ]:





# In[ ]:




