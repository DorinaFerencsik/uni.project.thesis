import pandas
import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

class MovingTimeRegression:
    def __init__(self, cleanedRoute, modelsRoute, scalerRoute, paramsRoute, cleanedFileName):   # define data routes and csv names!
        self.cleanedRoute = cleanedRoute
        self.modelsRoute = modelsRoute
        self.scalerRoute = scalerRoute
        self.paramsRoute = paramsRoute
        self.dataset = pandas.read_csv(cleanedRoute+cleanedFileName, sep="|")
        self.allTrainX = None
        self.allTestX = None
        self.allTrainy = None
        self.allTesty = None
        self.reducedTrainX = None
        self.reducedTestX = None
        self.reducedTrainy = None
        self.reducedTesty = None
        self.baseTrainX = None
        self.baseTestX = None
        self.baseTrainy = None
        self.baseTesty = None

        self.allRidgeModel = None
        self.reducedRidgeModel = None
        self.baseRidgeModel = None
        self.allLassoModel = None
        self.reducedLassoModel = None
        self.baseLassoModel = None
        self.allRandomForestModel = None
        self.reducedRandomForestModel = None
        self.baseRandomForestModel = None

        self.allStandardScaler = None
        self.reducedStandardScaler = None
        self.baseStandardScaler = None

        self.CatCols8 = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
        self.CatCols5 = ['#003f5c', '#3d61f4', '#bc5090', '#ff6361', '#ffa600']
        self.CatCols3 = ['#003f5c', '#bc5090', '#ffa600']
        print('call .dataset to see the cleaned dataset')

    def prepareData(self):
        # all data
        X = self.dataset.copy()
        Y = X['moving_time'].copy()
        X.drop(columns=['moving_time', 'elapsed_time', 'average_speed'], inplace=True)

        names = X.columns
        self.allStandardScaler = StandardScaler()

        scaledX = self.allStandardScaler.fit_transform(X)
        scaledX = pandas.DataFrame(scaledX, columns=names)

        self.allTrainX, self.allTestX, self.allTrainy, self.allTesty = train_test_split(scaledX, Y, random_state=42)

        # reduced data
        X = self.dataset[['age_0.0', 'age_1.0', 'age_2.0', 'distance', 'elev_high', 'elev_low', 'hashed_id',
                     'total_elevation_gain', 'trainer_onehot', 'workout_type_11.0', 'workout_type_10.0', 'workout_type_12.0']].copy()
        Y = self.dataset['moving_time'].copy()

        names = X.columns
        self.reducedStandardScaler = StandardScaler()

        scaledX = self.reducedStandardScaler.fit_transform(X)
        scaledX = pandas.DataFrame(scaledX, columns=names)

        self.reducedTrainX, self.reducedTestX, self.reducedTrainy, self.reducedTesty = train_test_split(scaledX, Y,
                                                                                    random_state=42)
        # base data
        X = self.dataset[['distance', 'elev_high', 'elev_low', 'total_elevation_gain', 'trainer_onehot']].copy()
        Y = self.dataset['moving_time'].copy()
        names = X.columns
        self.baseStandardScaler = StandardScaler()

        scaledX = self.baseStandardScaler.fit_transform(X)
        scaledX = pandas.DataFrame(scaledX, columns=names)

        self.baseTrainX, self.baseTestX, self.baseTrainy, self.baseTesty = train_test_split(scaledX, Y, random_state=42)
        # TODO: print info about the 3 dataset and the train-test pars

    def trainModels(self, verbose=False, writeToFile=False):
        # fitting models on all data
        print('-- Fitting models on all data -- ')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.allTrainX, self.allTestX, self.allTrainy, self.allTesty)
        self.allRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.allRidgeModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.allTrainX, self.allTestX, self.allTrainy, self.allTesty)
        self.allLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.allLassoModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        # TODO: calc best params for Random Forest
        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute+'moving_time_all_random_forest_params.p', 'rb'))
        self.allRandomForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                          max_features=forestParams['max_features'],
                                                          min_samples_leaf=forestParams['min_samples_leaf'] )
        self.allRandomForestModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')


        print('Scores on allTesty data: ')
        print(' - Ridge: '+str(self.allRidgeModel.score(self.allTestX, self.allTesty)))
        print(' - Lasso: '+str(self.allLassoModel.score(self.allTestX, self.allTesty)))
        print(' - RandomForest: '+str(self.allRandomForestModel.score(self.allTestX, self.allTesty)))
        print('')

        # fitting models on reduced data
        print('-- Fitting models on reduced data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.reducedTrainX, self.reducedTestX, self.reducedTrainy, self.reducedTesty)
        self.reducedRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.reducedRidgeModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.reducedTrainX, self.reducedTestX, self.reducedTrainy, self.reducedTesty)
        self.reducedLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.reducedLassoModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        # TODO: calc best params for Random Forest
        if verbose:
            print('Loading best params for RandomForest...')

        forestParams = pickle.load(open(self.paramsRoute + 'moving_time_reduced_random_forest_params.p', 'rb'))
        self.reducedRandomForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                          max_features=forestParams['max_features'],
                                                          min_samples_leaf=forestParams['min_samples_leaf'])
        self.reducedRandomForestModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on reudcedTesty data: ')
        print(' - Ridge: ' + str(self.reducedRidgeModel.score(self.reducedTestX, self.reducedTesty)))
        print(' - Lasso: ' + str(self.reducedLassoModel.score(self.reducedTestX, self.reducedTesty)))
        print(' - RandomForest: ' + str(self.reducedRandomForestModel.score(self.reducedTestX, self.reducedTesty)))
        print('')

        # fitting models on base data
        print('-- Fitting models on base data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.baseTrainX, self.baseTestX, self.baseTrainy,
                                            self.baseTesty)
        self.baseRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.baseRidgeModel.fit(self.baseTrainX, self.baseTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.baseTrainX, self.baseTestX, self.baseTrainy,
                                            self.baseTesty)
        self.baseLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.baseLassoModel.fit(self.baseTrainX, self.baseTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'moving_time_base_random_forest_params.p', 'rb'))
        self.baseRandomForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                              max_features=forestParams['max_features'],
                                                              min_samples_leaf=forestParams['min_samples_leaf'])
        self.baseRandomForestModel.fit(self.baseTrainX, self.baseTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on baseTesty data: ')
        print(' - Ridge: ' + str(self.baseRidgeModel.score(self.baseTestX, self.baseTesty)))
        print(' - Lasso: ' + str(self.baseLassoModel.score(self.baseTestX, self.baseTesty)))
        print(' - RandomForest: ' + str(self.baseRandomForestModel.score(self.baseTestX, self.baseTesty)))
        print('')

        if writeToFile:
            writeRegressionModels(self.allRidgeModel, self.allLassoModel, self.allRandomForestModel, 'all', 'moving',
                                  self.modelsRoute)
            writeRegressionModels(self.reducedRidgeModel, self.reducedLassoModel, self.reducedRandomForestModel, 'reduced',
                                  'moving', self.modelsRoute)
            writeRegressionModels(self.baseRidgeModel, self.baseLassoModel, self.baseRandomForestModel, 'base', 'moving',
                                  self.modelsRoute)
            writeScalerModel(self.allStandardScaler, 'all', 'moving', self.scalerRoute)
            writeScalerModel(self.reducedStandardScaler, 'reduced', 'moving', self.scalerRoute)
            writeScalerModel(self.baseStandardScaler, 'base', 'moving', self.scalerRoute)



        print('Get predictions: ')
        print(' - based on all data: call getPredictionWithAllModels(list) func., where list has 27 elements ')
        print(' - based on reduced data: call getPredictionWithReducedModels(list) func., where list has 9 elements ')
        print(' - based on all data: call getPredictionWithBaseModels(list) func., where list has 5 elements ')

    def calulateBestParamsForRandomForest(self):
        print('Warning: this function will take several hours: the results already available in the ./results/params folder')
        print('Consider interrupting the kernel')
        GridSearchForRandomForest(pandas.concat([self.allTrainX, self.allTestX]),
                                  pandas.concat([self.allTrainy, self.allTesty]), 'moving', 'all')
        GridSearchForRandomForest(pandas.concat([self.reducedTrainX, self.reducedTestX]),
                                  pandas.concat([self.reducedTrainy, self.reducedTesty]), 'moving', 'reduced')
        GridSearchForRandomForest(pandas.concat([self.baseTrainX, self.baseTestX]),
                                  pandas.concat([self.baseTrainy, self.baseTesty]), 'moving', 'base')

    def getPredictionWithAllModels(self, X):
        if len(X) != 26:
            print('Shape mismatch: X should contains 26 values like the allTrainX dataset')
            return
        #X.append(0.0)
        scaled = self.allStandardScaler.transform(np.reshape(X, (1,-1)))
        # TODO: scale X before calling predict func
        ridgeResult = self.allRidgeModel.predict(scaled)
        lassoResult = self.allLassoModel.predict(scaled)
        forestResult = self.allRandomForestModel.predict(scaled)
        print(' - ridge: '+ str(ridgeResult))
        print(' - lasso: '+ str(lassoResult))
        print(' - random forest: '+ str(forestResult))
        # TODO: create graph

    def getPredictionWithReducedModels(self, X):
        if len(X) != 9:
            print('Shape mismatch: X should contains 9 values like the reducedTrainX dataset')
            return
        # TODO: scale X before calling predict func
        scaled = self.reducedStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.reducedRidgeModel.predict(scaled)
        lassoResult = self.reducedLassoModel.predict(scaled)
        forestResult = self.reducedRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithBaseModels(self, X):
        if len(X) != 5:
            print('Shape mismatch: X should contains 5 values like the baseTrainX dataset')
            return
        # TODO: scale X before calling predict func
        scaled = self.baseStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.baseRidgeModel.predict(scaled)
        lassoResult = self.baseLassoModel.predict(scaled)
        forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def loadTrainedModelsAndScalers(self):
        self.loadRegressionModels()
        self.loadScalers()
        print('Regression models and scalers are loaded')
        print('Use the following functions to get predictions:')
        print(' - getPredictionWithAllModels')
        print(' - getPredictionWithReducedModels')
        print(' - getPredictionWithBaseModels')

    def loadRegressionModels(self):
        # loading models based on all dataset
        self.allRidgeModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'all' + '_' + 'ridge.p', 'rb'))
        self.allLassoModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'all' + '_' + 'lasso.p', 'rb'))
        self.allRandomForestModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'all' + '_' + 'random_forest.p', 'rb'))
        # loading models based on reduced dataset
        self.reducedRidgeModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'reduced' + '_' + 'ridge.p', 'rb'))
        self.reducedLassoModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'reduced' + '_' + 'lasso.p', 'rb'))
        self.reducedRandomForestModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'reduced' + '_' + 'random_forest.p', 'rb'))
        # loading models based on base dataset
        self.baseRidgeModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'base' + '_' + 'ridge.p', 'rb'))
        self.baseLassoModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'base' + '_' + 'lasso.p', 'rb'))
        self.baseRandomForestModel = pickle.load(
                    open(self.modelsRoute + 'moving_time_' + 'base' + '_' + 'random_forest.p', 'rb'))

    def loadScalers(self):
        # load fitted scaler models
        self.allStandardScaler = pickle.load(open(self.scalerRoute + 'moving_time_all_scaler.p', 'rb'))
        self.reducedStandardScaler = pickle.load(open(self.scalerRoute + 'moving_time_reduced_scaler.p', 'rb'))
        self.baseStandardScaler = pickle.load(open(self.scalerRoute + 'moving_time_base_scaler.p', 'rb'))

    def mps_to_kmph(self, m_per_s):
        return m_per_s * 3.6

    def kmph_to_mps(self, km_h):
        return km_h / 3.6
## END OF MOVING TIME CLASS

## ELAPSED TIME
class ElapsedTimeRegression():
    def __init__(self, cleanedRoute, modelsRoute, scalerRoute, paramsRoute, cleanedFileName):  # define data routes and csv names!
        self.cleanedRoute = cleanedRoute
        self.modelsRoute = modelsRoute
        self.scalerRoute = scalerRoute
        self.paramsRoute = paramsRoute
        self.dataset = pandas.read_csv(cleanedRoute + cleanedFileName, sep="|")
        # all data
        self.allTrainX = None
        self.allTestX = None
        self.allTrainy = None
        self.allTesty = None
        # reduced data
        self.reducedTrainX = None
        self.reducedTestX = None
        self.reducedTrainy = None
        self.reducedTesty = None

        # age group null data
        self.ageNullTrainX = None
        self.ageNullTestX = None
        self.ageNullTrainy = None
        self.ageNullTesty = None
        # age group one data
        self.ageOneTrainX = None
        self.ageOneTestX = None
        self.ageOneTrainy = None
        self.ageOneTesty = None
        # age group two data
        self.ageTwoTrainX = None
        self.ageTwoTestX = None
        self.ageTwoTrainy = None
        self.ageTwoTesty = None

        # distance small data
        self.distanceSmallTrainX = None
        self.distanceSmallTestX = None
        self.distanceSmallTrainy = None
        self.distanceSmallTesty = None
        # distance big data
        self.distanceBigTrainX = None
        self.distanceBigTestX = None
        self.distanceBigTrainy = None
        self.distanceBigTesty = None

        # user data
        self.userTrainX = None
        self.userTestX = None
        self.userTrainy = None
        self.userTesty = None

        # regression model initialization
        # ridge
        self.allRidgeModel = None
        self.reducedRidgeModel = None
        self.ageNullRidgeModel = None
        self.ageOneRidgeModel = None
        self.ageTwoRidgeModel = None
        self.distanceSmallRidgeModel = None
        self.distanceBigRidgeModel = None
        self.userRidgeModel = None
        # lasso
        self.allLassoModel = None
        self.reducedLassoModel = None
        self.ageNullLassoModel = None
        self.ageOneLassoModel = None
        self.ageTwoLassoModel = None
        self.distanceSmallLassoModel = None
        self.distanceBigLassoModel = None
        self.userLassoModel = None
        # random forest
        self.allForestModel = None
        self.reducedForestModel = None
        self.ageNullForestModel = None
        self.ageOneForestModel = None
        self.ageTwoForestModel = None
        self.distanceSmallForestModel = None
        self.distanceBigForestModel = None
        self.userForestModel = None

        self.allStandardScaler = None
        self.reducedStandardScaler = None
        self.ageNullStandardScaler = None
        self.ageOneStandardScaler = None
        self.ageTwoStandardScaler = None
        self.distanceSmallStandardScaler = None
        self.distanceBigStandardScaler = None
        self.userStandardScaler = None

        self.CatCols8 = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
        self.CatCols5 = ['#003f5c', '#3d61f4', '#bc5090', '#ff6361', '#ffa600']
        self.CatCols3 = ['#003f5c', '#bc5090', '#ffa600']
        print('call .dataset to see the cleaned dataset')

    def prepareData(self):
        # all data
        X = self.dataset.copy()
        Y = X['elapsed_time'].copy()
        X.drop(columns=['elapsed_time', 'average_speed'], inplace=True)

        names = X.columns
        self.allStandardScaler = StandardScaler()
        scaledX = self.allStandardScaler.fit_transform(X)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.allTrainX, self.allTestX, self.allTrainy, self.allTesty = train_test_split(scaledX, Y, random_state=42)

        # reduced data
        X = self.dataset[['age_0.0', 'age_1.0', 'age_2.0', 'distance', 'elev_high', 'elev_low', 'hashed_id',
                     'total_elevation_gain', 'moving_time', 'trainer_onehot', 'workout_type_11.0', 'workout_type_10.0', 'workout_type_12.0']].copy()
        Y = self.dataset['elapsed_time'].copy()

        names = X.columns
        self.reducedStandardScaler = StandardScaler()
        scaledX = self.reducedStandardScaler.fit_transform(X)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.reducedTrainX, self.reducedTestX, self.reducedTrainy, self.reducedTesty = train_test_split(scaledX, Y,
                                                                                    random_state=42)
        # age group: null data
        ageNull = self.dataset[self.dataset['age_0.0'] == 1].copy()
        Y = ageNull['elapsed_time'].copy()
        ageNull.drop(columns=['age_0.0', 'age_1.0', 'age_2.0', 'elapsed_time', 'average_speed'], inplace=True)

        names = ageNull.columns
        self.ageNullStandardScaler = StandardScaler()
        scaledX = self.ageNullStandardScaler.fit_transform(ageNull)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.ageNullTrainX, self.ageNullTestX, self.ageNullTrainy, self.ageNullTesty = train_test_split(scaledX, Y,
                                                                                                        random_state=42)
        # age group: one data
        ageOne = self.dataset[self.dataset['age_1.0'] == 1].copy()
        Y = ageOne['elapsed_time'].copy()
        ageOne.drop(columns=['age_0.0', 'age_1.0', 'age_2.0', 'elapsed_time', 'average_speed'], inplace=True)

        names = ageOne.columns
        self.ageOneStandardScaler = StandardScaler()
        scaledX = self.ageNullStandardScaler.fit_transform(ageOne)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.ageOneTrainX, self.ageOneTestX, self.ageOneTrainy, self.ageOneTesty = train_test_split(scaledX, Y,
                                                                                                        random_state=42)
        # age group: two data
        ageTwo = self.dataset[self.dataset['age_2.0'] == 1].copy()
        Y = ageTwo['elapsed_time'].copy()
        ageTwo.drop(columns=['age_0.0', 'age_1.0', 'age_2.0', 'elapsed_time', 'average_speed'], inplace=True)

        names = ageTwo.columns
        self.ageTwoStandardScaler = StandardScaler()
        scaledX = self.ageTwoStandardScaler.fit_transform(ageTwo)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.ageTwoTrainX, self.ageTwoTestX, self.ageTwoTrainy, self.ageTwoTesty = train_test_split(scaledX, Y,
                                                                                                        random_state=42)
        # distance small data
        distanceSmall = self.dataset[self.dataset['distance'] < 50000].copy()
        Y = distanceSmall['elapsed_time'].copy()
        distanceSmall.drop(columns=['elapsed_time', 'average_speed'], inplace=True)
        distanceSmall.reset_index(drop=True, inplace=True)

        names = distanceSmall.columns
        self.distanceSmallStandardScaler = StandardScaler()
        scaledX = self.distanceSmallStandardScaler.fit_transform(distanceSmall)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.distanceSmallTrainX, self.distanceSmallTestX, self.distanceSmallTrainy, self.distanceSmallTesty = train_test_split(scaledX, Y,
                                                                                                    random_state=42)
        # distance big data
        distanceBig = self.dataset[self.dataset['distance'] >= 50000].copy()
        Y = distanceBig['elapsed_time'].copy()
        distanceBig.drop(columns=['elapsed_time', 'average_speed'], inplace=True)
        distanceBig.reset_index(drop=True, inplace=True)

        names = distanceBig.columns
        self.distanceBigStandardScaler = StandardScaler()
        scaledX = self.distanceBigStandardScaler.fit_transform(distanceBig)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.distanceBigTrainX, self.distanceBigTestX, self.distanceBigTrainy, self.distanceBigTesty = train_test_split(
            scaledX, Y,
            random_state=42)

        # user with the most activities
        userData = self.dataset[self.dataset['hashed_id'] == self.dataset['hashed_id'].value_counts().idxmax()]
        Y = userData['elapsed_time'].copy()
        userData.drop(columns=['age_0.0', 'age_1.0', 'age_2.0', 'elapsed_time', 'average_speed'], inplace=True)

        names = userData.columns
        self.userStandardScaler = StandardScaler()
        scaledX = self.userStandardScaler.fit_transform(userData)
        scaledX = pandas.DataFrame(scaledX, columns=names)
        self.userTrainX, self.userTestX, self.userTrainy, self.userTesty = train_test_split(scaledX, Y, random_state=42)

        # TODO: user based dataset
        # TODO: print info about the 3 dataset and the train-test pars

    def trainModels(self, verbose=False, writeToFile=False):
        # fitting models on all data
        print('-- Fitting models on all data -- ')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.allTrainX, self.allTestX, self.allTrainy, self.allTesty)
        self.allRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.allRidgeModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.allTrainX, self.allTestX, self.allTrainy, self.allTesty)
        self.allLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.allLassoModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_all_random_forest_params.p', 'rb'))
        self.allForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                              max_features=forestParams['max_features'],
                                                              min_samples_leaf=forestParams['min_samples_leaf'])
        self.allForestModel.fit(self.allTrainX, self.allTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on allTesty data: ')
        print(' - Ridge: ' + str(self.allRidgeModel.score(self.allTestX, self.allTesty)))
        print(' - Lasso: ' + str(self.allLassoModel.score(self.allTestX, self.allTesty)))
        print(' - Random Forest: ' + str(self.allForestModel.score(self.allTestX, self.allTesty)))
        print('')

        # fitting models on reduced data
        print('-- Fitting models on reduced data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.reducedTrainX, self.reducedTestX, self.reducedTrainy,
                                            self.reducedTesty)
        self.reducedRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.reducedRidgeModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.reducedTrainX, self.reducedTestX, self.reducedTrainy,
                                            self.reducedTesty)
        self.reducedLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.reducedLassoModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_reduced_random_forest_params.p', 'rb'))
        self.reducedForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                    max_features=forestParams['max_features'],
                                                    min_samples_leaf=forestParams['min_samples_leaf'])
        self.reducedForestModel.fit(self.reducedTrainX, self.reducedTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on reducedTesty data: ')
        print(' - Ridge: ' + str(self.reducedRidgeModel.score(self.reducedTestX, self.reducedTesty)))
        print(' - Lasso: ' + str(self.reducedLassoModel.score(self.reducedTestX, self.reducedTesty)))
        print(' - Random Forest: ' + str(self.reducedForestModel.score(self.reducedTestX, self.reducedTesty)))
        print('')

        # fitting models on age group NULL data
        print('-- Fitting models on age group null data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.ageNullTrainX, self.ageNullTestX, self.ageNullTrainy,
                                            self.ageNullTesty)
        self.ageNullRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.ageNullRidgeModel.fit(self.ageNullTrainX, self.ageNullTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.ageNullTrainX, self.ageNullTestX, self.ageNullTrainy,
                                            self.ageNullTesty)
        self.ageNullLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.ageNullLassoModel.fit(self.ageNullTrainX, self.ageNullTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_age_null_random_forest_params.p', 'rb'))
        self.ageNullForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.ageNullForestModel.fit(self.ageNullTrainX, self.ageNullTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on age group null data: ')
        print(' - Ridge: ' + str(self.ageNullRidgeModel.score(self.ageNullTestX, self.ageNullTesty)))
        print(' - Lasso: ' + str(self.ageNullLassoModel.score(self.ageNullTestX, self.ageNullTesty)))
        print(' - Random Forest: ' + str(self.ageNullForestModel.score(self.ageNullTestX, self.ageNullTesty)))
        print('')

        # fitting models on age group ONE data
        print('-- Fitting models on age group one data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.ageOneTrainX, self.ageOneTestX, self.ageOneTrainy,
                                            self.ageOneTesty)
        self.ageOneRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.ageOneRidgeModel.fit(self.ageOneTrainX, self.ageOneTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.ageOneTrainX, self.ageOneTestX, self.ageOneTrainy,
                                            self.ageOneTesty)
        self.ageOneLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.ageOneLassoModel.fit(self.ageOneTrainX, self.ageOneTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_age_one_random_forest_params.p', 'rb'))
        self.ageOneForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.ageOneForestModel.fit(self.ageOneTrainX, self.ageOneTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on age group one data: ')
        print(' - Ridge: ' + str(self.ageOneRidgeModel.score(self.ageOneTestX, self.ageOneTesty)))
        print(' - Lasso: ' + str(self.ageOneLassoModel.score(self.ageOneTestX, self.ageOneTesty)))
        print(' - Random Forest: ' + str(self.ageOneLassoModel.score(self.ageOneTestX, self.ageOneTesty)))
        print('')

        # fitting models on age group TWO data
        print('-- Fitting models on age group two data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.ageTwoTrainX, self.ageTwoTestX, self.ageTwoTrainy,
                                            self.ageTwoTesty)
        self.ageTwoRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.ageTwoRidgeModel.fit(self.ageTwoTrainX, self.ageTwoTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.ageTwoTrainX, self.ageTwoTestX, self.ageTwoTrainy,
                                            self.ageTwoTesty)
        self.ageTwoLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.ageTwoLassoModel.fit(self.ageTwoTrainX, self.ageTwoTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_age_two_random_forest_params.p', 'rb'))
        self.ageTwoForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.ageTwoForestModel.fit(self.ageTwoTrainX, self.ageTwoTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on age group two data: ')
        print(' - Ridge: ' + str(self.ageTwoRidgeModel.score(self.ageTwoTestX, self.ageTwoTesty)))
        print(' - Lasso: ' + str(self.ageTwoLassoModel.score(self.ageTwoTestX, self.ageTwoTesty)))
        print(' - Random Forest: ' + str(self.ageTwoForestModel.score(self.ageTwoTestX, self.ageTwoTesty)))
        print('')

        # fitting models on distance small data
        print('-- Fitting models on distance small data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.distanceSmallTrainX, self.distanceSmallTestX, self.distanceSmallTrainy,
                                            self.distanceSmallTesty)
        self.distanceSmallRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.distanceSmallRidgeModel.fit(self.distanceSmallTrainX, self.distanceSmallTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.distanceSmallTrainX, self.distanceSmallTestX, self.distanceSmallTrainy,
                                            self.distanceSmallTesty)
        self.distanceSmallLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.distanceSmallLassoModel.fit(self.distanceSmallTrainX, self.distanceSmallTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_distance_small_random_forest_params.p', 'rb'))
        self.distanceSmallForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.distanceSmallForestModel.fit(self.distanceSmallTrainX, self.distanceSmallTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on distanceSmall data: ')
        print(' - Ridge: ' + str(self.distanceSmallRidgeModel.score(self.distanceSmallTestX, self.distanceSmallTesty)))
        print(' - Lasso: ' + str(self.distanceSmallLassoModel.score(self.distanceSmallTestX, self.distanceSmallTesty)))
        print(' - Random Forest: ' + str(self.distanceSmallForestModel.score(self.distanceSmallTestX, self.distanceSmallTesty)))
        print('')

        # fitting models on distance big data
        print('-- Fitting models on distance big data --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.distanceBigTrainX, self.distanceBigTestX, self.distanceBigTrainy,
                                            self.distanceBigTesty)
        self.distanceBigRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.distanceBigRidgeModel.fit(self.distanceBigTrainX, self.distanceBigTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.distanceBigTrainX, self.distanceBigTestX, self.distanceBigTrainy,
                                            self.distanceBigTesty)
        self.distanceBigLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.distanceBigLassoModel.fit(self.distanceBigTrainX, self.distanceBigTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_distance_big_random_forest_params.p', 'rb'))
        self.distanceBigForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.distanceBigForestModel.fit(self.distanceBigTrainX, self.distanceBigTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on distanceBig data: ')
        print(' - Ridge: ' + str(self.distanceBigRidgeModel.score(self.distanceBigTestX, self.distanceBigTesty)))
        print(' - Lasso: ' + str(self.distanceBigLassoModel.score(self.distanceBigTestX, self.distanceBigTesty)))
        print(' - Random Forest: ' + str(self.distanceBigForestModel.score(self.distanceBigTestX, self.distanceBigTesty)))
        print('')

        # fitting models on user data
        print('-- Fitting models on user data (with the most activities -'+str(len(self.userTestX) + len(self.userTrainX))+') --')
        if verbose:
            print('Calculating best params for Ridge...')
        ridgeParams = getBestParamsForRidge(self.userTrainX, self.userTestX, self.userTrainy, self.userTesty)
        self.userRidgeModel = Ridge(alpha=ridgeParams['alpha'], solver=ridgeParams['solver'])
        self.userRidgeModel.fit(self.userTrainX, self.userTrainy)
        if verbose:
            print('Done. Params: ')
            print(ridgeParams)
            print('')

        if verbose:
            print('Calculating best params for Lasso...')
        lassoParams = getBestParamsForLasso(self.userTrainX, self.userTestX, self.userTrainy, self.userTesty)
        self.userLassoModel = Lasso(alpha=lassoParams['alpha'])
        self.userLassoModel.fit(self.userTrainX, self.userTrainy)
        if verbose:
            print('Done. Params: ')
            print(lassoParams)
            print('')

        if verbose:
            print('Loading best params for RandomForest...')
        forestParams = pickle.load(open(self.paramsRoute + 'elapsed_time_user_random_forest_params.p', 'rb'))
        self.userForestModel = RandomForestRegressor(n_estimators=forestParams['n_estimators'],
                                                        max_features=forestParams['max_features'],
                                                        min_samples_leaf=forestParams['min_samples_leaf'])
        self.userForestModel.fit(self.userTrainX, self.userTrainy)
        if verbose:
            print('Done. Params: ')
            print(forestParams)
            print('')

        print('Scores on user data: ')
        print(' - Ridge: ' + str(self.userRidgeModel.score(self.userTestX, self.userTesty)))
        print(' - Lasso: ' + str(self.userLassoModel.score(self.userTestX, self.userTesty)))
        print(' - Random Forest: ' + str(self.userForestModel.score(self.userTestX, self.userTesty)))
        print('')


        # write models and scalers to pickle file
        if writeToFile:
            # write regression models
            writeRegressionModels(self.allRidgeModel, self.allLassoModel, None, 'all',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.reducedRidgeModel, self.reducedLassoModel, None, 'reduced',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.ageNullRidgeModel, self.ageNullLassoModel, None, 'ageNull',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.ageOneRidgeModel, self.ageOneLassoModel, None, 'ageOne',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.ageTwoRidgeModel, self.ageTwoLassoModel, None, 'ageTwo',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.distanceSmallRidgeModel, self.distanceSmallLassoModel, None, 'distanceSmall',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.distanceBigRidgeModel, self.distanceBigLassoModel, None, 'distanceBig',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)
            writeRegressionModels(self.userRidgeModel, self.userLassoModel, None, 'user',
                                  'elapsed', self.modelsRoute, hasRandomForest=False)

            # write scalers
            writeScalerModel(self.allStandardScaler, 'all', 'elapsed', self.scalerRoute)
            writeScalerModel(self.reducedStandardScaler, 'reduced', 'elapsed', self.scalerRoute)
            writeScalerModel(self.ageNullStandardScaler, 'ageNull', 'elapsed', self.scalerRoute)
            writeScalerModel(self.ageOneStandardScaler, 'ageOne', 'elapsed', self.scalerRoute)
            writeScalerModel(self.ageTwoStandardScaler, 'ageTwo', 'elapsed', self.scalerRoute)
            writeScalerModel(self.distanceSmallStandardScaler, 'distanceSmall', 'elapsed', self.scalerRoute)
            writeScalerModel(self.distanceBigStandardScaler, 'distanceBig', 'elapsed', self.scalerRoute)
            writeScalerModel(self.userStandardScaler, 'user', 'elapsed', self.scalerRoute)

        print('Get predictions: ')
        print(' - based on all data: call getPredictionWithAllModels(list) func., where list has 27 elements ')
        print(' - based on reduced data: call getPredictionWithReducedModels(list) func., where list has 9 elements ')
        # print(' - based on base data: call getPredictionWithBaseModels(list) func., where list has 5 elements ')

    def getPredictionWithAllModels(self, X):
        if len(X) != 27:
            print('Shape mismatch: X should contains 27 values like the allTrainX dataset')
            return
        scaled = self.allStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.allRidgeModel.predict(scaled)
        lassoResult = self.allLassoModel.predict(scaled)
        #forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        #print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithReducedModels(self, X):
        if len(X) != 10:
            print('Shape mismatch: X should contains 10 values like the reducedTrainX dataset')
            return
        scaled = self.reducedStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.reducedRidgeModel.predict(scaled)
        lassoResult = self.reducedLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithAgeNullModels(self, X):
        if len(X) != 24:
            print('Shape mismatch: X should contains 24 values like the ageNullTrainX dataset')
            return
        scaled = self.ageNullStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.ageNullRidgeModel.predict(scaled)
        lassoResult = self.ageNullLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithAgeOneModels(self, X):
        if len(X) != 24:
            print('Shape mismatch: X should contains 24 values like the ageOneTrainX dataset')
            return
        # TODO: scale X before calling predict func
        scaled = self.ageOneStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.ageOneRidgeModel.predict(scaled)
        lassoResult = self.ageOneLassoModel.predict(scaled)
        # forestResult = self.ageOneRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithAgeTwoModels(self, X):
        if len(X) != 24:
            print('Shape mismatch: X should contains 24 values like the ageTwoTrainX dataset')
            return
        # TODO: scale X before calling predict func
        scaled = self.ageTwoStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.ageTwoRidgeModel.predict(scaled)
        lassoResult = self.ageTwoLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithDistanceSmallModels(self, X):
        if len(X) != 27:
            print('Shape mismatch: X should contains 27 values like the distanceSmallTrainX dataset')
            return
        # TODO: scale X before calling predict func
        scaled = self.distanceSmallStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.distanceSmallRidgeModel.predict(scaled)
        lassoResult = self.distanceSmallLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithDistanceBigModels(self, X):
        if len(X) != 27:
            print('Shape mismatch: X should contains 27 values like the distanceBigTrainX dataset')
            return
        scaled = self.distanceBigStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.distanceBigRidgeModel.predict(scaled)
        lassoResult = self.distanceBigLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def getPredictionWithUserModels(self, X):
        if len(X) != 24:
            print('Shape mismatch: X should contains 24 values like the userTrainX dataset')
            return
        scaled = self.userStandardScaler.transform(np.reshape(X, (1,-1)))
        ridgeResult = self.userRidgeModel.predict(scaled)
        lassoResult = self.userLassoModel.predict(scaled)
        # forestResult = self.baseRandomForestModel.predict(scaled)
        print(' - ridge: ' + str(ridgeResult))
        print(' - lasso: ' + str(lassoResult))
        # print(' - random forest: ' + str(forestResult))
        # TODO: create graph

    def calulateBestParamsForRandomForest(self):
        print('Warning: this function will take several hours: the results already available in the ./data/results/params folder')
        print('Consider interrupting the kernel')
        GridSearchForRandomForest(pandas.concat([self.allTrainX, self.allTestX]),
                                  pandas.concat([self.allTrainy, self.allTesty]), 'elapsed', 'all')
        GridSearchForRandomForest(pandas.concat([self.reducedTrainX, self.reducedTestX]),
                                  pandas.concat([self.reducedTrainy, self.reducedTesty]), 'elapsed', 'reduced')
        GridSearchForRandomForest(pandas.concat([self.ageNullTrainX, self.ageNullTestX]),
                                  pandas.concat([self.ageNullTrainy, self.ageNullTesty]), 'elapsed', 'age_null')
        GridSearchForRandomForest(pandas.concat([self.ageOneTrainX, self.ageOneTestX]),
                                  pandas.concat([self.ageOneTrainy, self.ageOneTesty]), 'elapsed', 'age_one')
        GridSearchForRandomForest(pandas.concat([self.ageTwoTrainX, self.ageTwoTestX]),
                                  pandas.concat([self.ageTwoTrainy, self.ageTwoTesty]), 'elapsed', 'age_two')
        GridSearchForRandomForest(pandas.concat([self.distanceSmallTrainX, self.distanceSmallTestX]),
                                  pandas.concat([self.distanceSmallTrainy, self.distanceSmallTesty]), 'elapsed', 'distance_small')
        GridSearchForRandomForest(pandas.concat([self.distanceBigTrainX, self.distanceBigTestX]),
                                  pandas.concat([self.distanceBigTrainy, self.distanceBigTesty]), 'elapsed', 'distance_big')
        GridSearchForRandomForest(pandas.concat([self.userTrainX, self.userTestX]),
                                  pandas.concat([self.userTrainy, self.userTesty]), 'elapsed', 'user')

        # self.allRidgeModel = None
        # self.reducedRidgeModel = None
        # self.ageNullRidgeModel = None
        # self.ageOneRidgeModel = None
        # self.ageTwoRidgeModel = None
        # self.distanceSmallRidgeModel = None
        # self.distanceBigRidgeModel = None
        # self.userRidgeModel = None

    def loadTrainedModelsAndScalers(self):
        self.loadRegressionModels()
        self.loadScalers()
        print('Regression models and scalers are loaded')
        print('Use the following functions to get predictions:')
        print(' - getPredictionWithAllModels')
        print(' - getPredictionWithReducedModels')
        print(' - getPredictionWithAgeNullModels')
        print(' - getPredictionWithAgeOneModels')
        print(' - getPredictionWithDistanceSmallModels')
        print(' - getPredictionWithDistanceBigModels')
        print(' - getPredictionWithUserModels')

    def loadRegressionModels(self):
        # loading models based on all dataset
        self.allRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'all' + '_' + 'ridge.p', 'rb'))
        self.allLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'all' + '_' + 'lasso.p', 'rb'))

        # loading models based on reduced dataset
        self.reducedRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'reduced' + '_' + 'ridge.p', 'rb'))
        self.reducedLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'reduced' + '_' + 'lasso.p', 'rb'))
        # loading models based on age null dataset
        self.ageNullRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageNull' + '_' + 'ridge.p', 'rb'))
        self.ageNullLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageNull' + '_' + 'lasso.p', 'rb'))
        # loading models based on age one dataset
        self.ageOneRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageOne' + '_' + 'ridge.p', 'rb'))
        self.ageOneLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageOne' + '_' + 'lasso.p', 'rb'))
        # loading models based on age two dataset
        self.ageTwoRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageTwo' + '_' + 'ridge.p', 'rb'))
        self.ageTwoLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'ageTwo' + '_' + 'lasso.p', 'rb'))
        # loading models based on distance small dataset
        self.distanceSmallRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'distanceSmall' + '_' + 'ridge.p', 'rb'))
        self.distanceSmallLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'distanceSmall' + '_' + 'lasso.p', 'rb'))
        # loading models based on distance big dataset
        self.distanceBigRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'distanceBig' + '_' + 'ridge.p', 'rb'))
        self.distanceBigLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'distanceBig' + '_' + 'lasso.p', 'rb'))
        # loading models based on user dataset
        self.userRidgeModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'user' + '_' + 'ridge.p', 'rb'))
        self.userLassoModel = pickle.load(
                    open(self.modelsRoute + 'elapsed_time_' + 'user' + '_' + 'lasso.p', 'rb'))

    def loadScalers(self):
        # load fitted scaler models
        self.allStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_all_scaler.p', 'rb'))
        self.reducedStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_reduced_scaler.p', 'rb'))
        self.ageNullStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_ageNull_scaler.p', 'rb'))
        self.ageOneStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_ageOne_scaler.p', 'rb'))
        self.ageTwoStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_ageTwo_scaler.p', 'rb'))
        self.distanceSmallStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_distanceSmall_scaler.p', 'rb'))
        self.distanceBigStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_distanceBig_scaler.p', 'rb'))
        self.userStandardScaler = pickle.load(open(self.scalerRoute + 'elapsed_time_user_scaler.p', 'rb'))

    def mps_to_kmph(self, m_per_s):
        return m_per_s * 3.6

    def kmph_to_mps(self, km_h):
        return km_h / 3.6

## END OF ELAPSED TIME CLASS
def GridSearchForRandomForest(X, y, timeType, dataType):
    n_estimators = [200, 500, 1000]
    max_features = ['auto', 'sqrt']
    min_samples_leaf = [1, 5, 10, 30, 60]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'min_samples_leaf': min_samples_leaf}
    clf = GridSearchCV(RandomForestRegressor(), random_grid, cv=5, n_jobs=3)
    clf.fit(X, y)

    pickle.dump(clf.best_params_, open('./results/params/' + str(timeType) + '_time_' + str(dataType) + '_' + 'random_forest_params.p', 'wb'))
    print('Params for '+str(timeType) + ' time and data: ' + str(dataType) + ' written to pickle file')

def writeRegressionModels(ridge, lasso, randomForest, dataType, timeType, route, hasRandomForest=True):
    if timeType == 'moving' or timeType == 'elapsed':
        print('Writing model pickle file for '+timeType+'_time based on '+dataType+' data')
        # filename prefix: time_type_data_model.p
        # for example: moving_time_all_ridge.p
        pickle.dump(ridge, open(route + timeType + '_time_' + dataType + '_' + 'ridge.p', 'wb'))
        pickle.dump(lasso, open(route + timeType + '_time_' + dataType + '_' + 'lasso.p', 'wb'))
        if hasRandomForest:
            pickle.dump(randomForest, open(route + timeType + '_time_' + dataType + '_' + 'random_forest.p', 'wb'))
    else:
        print('ERROR: invalid timeType. Use "moving" or "elapsed" instead')

def writeScalerModel(scaler, dataType, timeType, route):
    if timeType == 'moving' or timeType == 'elapsed':
        print('Writing scaler pickle file for '+timeType+'_time based on '+dataType+' data')
        pickle.dump(scaler, open(route + timeType + '_time_' + dataType + '_scaler.p', 'wb'))
    else:
        print('ERROR: invalid timeType. Use "moving" or "elapsed" instead')

def getBestParamsForRidge(trainX, testX, trainy, testy):
    alpha = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 0.3, 0.35, 0.6]
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    bestScore = 0.0
    bestAlpha = 0.0
    bestSolver = ''
    for a in alpha:
        for s in solver:
            iterations = 10
            iterScore = 0
            for i in range(0,iterations):
                ridgeModel = Ridge(alpha=a, solver=s)
                ridgeModel.fit(trainX, trainy)
                iterScore = iterScore + ridgeModel.score(testX, testy)
            iterScore = iterScore / iterations
            if iterScore > bestScore:
                bestScore = iterScore
                bestAlpha = a
                bestSolver = s
    #print('Done')
    return {'alpha': bestAlpha, 'solver': bestSolver, 'score': bestScore}


def getBestParamsForLasso(trainX, testX, trainy, testy):
    alpha = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 0.3, 0.35, 0.6]
    bestScore = 0.0
    bestAlpha = 0.0
    for a in alpha:
        lassoModel = Lasso(alpha=a)
        lassoModel.fit(trainX, trainy)
        if lassoModel.score(testX, testy) > bestScore:
            bestScore = lassoModel.score(testX, testy)
            bestAlpha = a

    #print("Done")
    return {'alpha': bestAlpha, 'score': bestScore}
