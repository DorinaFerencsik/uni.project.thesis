import pandas
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, MeanShift, SpectralClustering, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.nearest_centroid import NearestCentroid




class DifficultyLevelClustering:
    def __init__(self, cleanedRoute, modelsRoute, resultDataRoute, cleanedFileName):
        self.cleanedRoute = cleanedRoute
        self.modelsRoute = modelsRoute
        self.resultDataRoute = resultDataRoute
        self.dataset = pandas.read_csv(cleanedRoute + cleanedFileName, sep="|")

        self.ageNullScaled = None
        self.ageNullInverted = None
        self.ageNullKmeans = None
        self.ageNullMiniBatch = None
        self.ageNullSpectral = None

        self.ageOneScaled = None
        self.ageOneInverted = None
        self.ageOneKmeans = None
        self.ageOneMiniBatch = None
        self.ageOneSpectral = None

        self.ageTwoScaled = None
        self.ageTwoInverted = None
        self.ageTwoKmeans = None
        self.ageTwoMiniBatch = None
        self.ageTwoSpectral = None

        self.ageNullStandardScaler = None
        self.ageOneStandardScaler = None
        self.ageTwoStandardScaler = None

        self.CatCols8 = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
        self.CatCols5 = ['#003f5c', '#3d61f4', '#bc5090', '#ff6361', '#ffa600']
        self.CatCols3 = ['#003f5c', '#bc5090', '#ffa600']

    def prepareData(self):
        # age group null
        data = self.dataset[self.dataset['age_0.0'] == 1].copy()
        data = data[data['distance'] < 99000].copy()
        reducedData = data[['average_speed', 'distance']].copy()

        self.ageNullStandardScaler = StandardScaler()
        self.ageNullScaled = self.ageNullStandardScaler.fit_transform(reducedData)
        self.ageNullScaled = pandas.DataFrame(self.ageNullScaled, columns=reducedData.columns)

        self.ageNullInverted = self.ageNullStandardScaler.inverse_transform(self.ageNullScaled)
        self.ageNullInverted = pandas.DataFrame(self.ageNullInverted, columns=reducedData.columns)

        # age group one
        data = self.dataset[self.dataset['age_1.0'] == 1].copy()
        data = data[data['distance'] < 99000].copy()
        reducedData = data[['average_speed', 'distance']].copy()

        self.ageOneStandardScaler = StandardScaler()
        self.ageOneScaled = self.ageOneStandardScaler.fit_transform(reducedData)
        self.ageOneScaled = pandas.DataFrame(self.ageOneScaled, columns=reducedData.columns)

        self.ageOneInverted = self.ageOneStandardScaler.inverse_transform(self.ageOneScaled)
        self.ageOneInverted = pandas.DataFrame(self.ageOneInverted, columns=reducedData.columns)

        # age group two
        data = self.dataset[self.dataset['age_2.0'] == 1].copy()
        data = data[data['distance'] < 99000].copy()
        reducedData = data[['average_speed', 'distance']].copy()

        self.ageTwoStandardScaler = StandardScaler()
        self.ageTwoScaled = self.ageTwoStandardScaler.fit_transform(reducedData)
        self.ageTwoScaled = pandas.DataFrame(self.ageTwoScaled, columns=reducedData.columns)

        self.ageTwoInverted = self.ageTwoStandardScaler.inverse_transform(self.ageTwoScaled)
        self.ageTwoInverted = pandas.DataFrame(self.ageTwoInverted, columns=reducedData.columns)

    def trainAgeNullModels(self, verbose=False, writeToFile=False, graphLang='en'):
        # KMeans model
        if verbose:
            print('training KMeans model')
        self.ageNullKmeans = KMeans(n_clusters=5)
        self.ageNullKmeans.fit(self.ageNullScaled)

        self.ageNullInverted['kmeans'] = self.ageNullKmeans.labels_

        # MiniBatch KMeans model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageNullMiniBatch = MiniBatchKMeans(n_clusters=5, random_state=0)
        self.ageNullMiniBatch.fit(self.ageNullScaled)

        self.ageNullInverted['minibatch'] = self.ageNullMiniBatch.labels_

        # Spectral model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageNullSpectral = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0)
        self.ageNullSpectral.fit(self.ageNullScaled)

        self.ageNullInverted['spectral'] = self.ageNullSpectral.labels_

        if writeToFile:
            writeDataFile(self.ageNullInverted, '01234', 0, self.resultDataRoute)
            writeModels(self.ageNullKmeans, self.ageNullMiniBatch, self.ageNullSpectral, 0, self.modelsRoute)

        # map cluster centers to A-E values
        kmeansDict = getAgeNullMapping(self.ageNullKmeans.cluster_centers_)
        self.ageNullInverted.replace({'kmeans': kmeansDict}, inplace=True)

        miniBatchDict = getAgeNullMapping(self.ageNullMiniBatch.cluster_centers_)
        self.ageNullInverted.replace({'minibatch': miniBatchDict}, inplace=True)

        nearestCentrid = NearestCentroid()
        nearestCentrid.fit(self.ageNullScaled, self.ageNullInverted['spectral'])

        spectralDict = getAgeNullMapping(nearestCentrid.centroids_)
        self.ageNullInverted.replace({'spectral': spectralDict}, inplace=True)

        if writeToFile:
            writeDataFile(self.ageNullInverted, 'ABCDE', 0, self.resultDataRoute)

        createClusterResultsPlot(self.ageNullInverted, self.CatCols5, 0, writeToFile, graphLang)
        print(' -- Cluster information: --')
        print(' - A : norm to high distance - norm speed')
        print(' - B : high distance - norm to high speed')
        print(' - C : ext. high distance - high to ext. high speed')
        print(' - D : norm distance - high speed')
        print(' - E : norm to high distance - ext. high speed')

    def trainAgeOneModels(self, verbose=False, writeToFile=False, graphLang='en'):
        # KMeans model
        if verbose:
            print('training KMeans model')
        self.ageOneKmeans = KMeans(n_clusters=5)
        self.ageOneKmeans.fit(self.ageOneScaled)

        self.ageOneInverted['kmeans'] = self.ageOneKmeans.labels_

        # MiniBatch KMeans model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageOneMiniBatch = MiniBatchKMeans(n_clusters=5, random_state=0)
        self.ageOneMiniBatch.fit(self.ageOneScaled)

        self.ageOneInverted['minibatch'] = self.ageOneMiniBatch.labels_

        # Spectral model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageOneSpectral = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0)
        self.ageOneSpectral.fit(self.ageOneScaled)

        self.ageOneInverted['spectral'] = self.ageOneSpectral.labels_

        if writeToFile:
            writeDataFile(self.ageOneInverted, '01234', 1, self.resultDataRoute)
            writeModels(self.ageOneKmeans, self.ageOneMiniBatch, self.ageOneSpectral, 1, self.modelsRoute)

        # map cluster centers to A-E values
        kmeansDict = getAgeOneMapping(self.ageOneKmeans.cluster_centers_)
        self.ageOneInverted.replace({'kmeans': kmeansDict}, inplace=True)

        miniBatchDict = getAgeOneMapping(self.ageOneMiniBatch.cluster_centers_)
        self.ageOneInverted.replace({'minibatch': miniBatchDict}, inplace=True)

        nearestCentrid = NearestCentroid()
        nearestCentrid.fit(self.ageOneScaled, self.ageOneInverted['spectral'])

        spectralDict = getAgeOneMapping(nearestCentrid.centroids_)
        self.ageOneInverted.replace({'spectral': spectralDict}, inplace=True)

        if writeToFile:
            writeDataFile(self.ageOneInverted, 'ABCDE', 1, self.resultDataRoute)

        createClusterResultsPlot(self.ageOneInverted, self.CatCols5, 1, writeToFile, graphLang)
        print('-- Cluster info --')
        print(' - A : norm to high distance - norm to ext. high speed')
        print(' - B : high distance - norm speed')
        print(' - C : ext. high distance - norm to ext. high speed')
        print(' - D : norm distance - high speed')
        print(' - E : norm to high distance - ext. high speed')


    def trainAgeTwoModels(self, verbose=False, writeToFile=False, graphLang='en'):
        # KMeans model
        if verbose:
            print('training KMeans model')
        self.ageTwoKmeans = KMeans(n_clusters=5)
        self.ageTwoKmeans.fit(self.ageTwoScaled)

        self.ageTwoInverted['kmeans'] = self.ageTwoKmeans.labels_

        # MiniBatch KMeans model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageTwoMiniBatch = MiniBatchKMeans(n_clusters=5, random_state=0)
        self.ageTwoMiniBatch.fit(self.ageTwoScaled)

        self.ageTwoInverted['minibatch'] = self.ageTwoMiniBatch.labels_

        # Spectral model
        if verbose:
            print('training MiniBatch KMeans model')
        self.ageTwoSpectral = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0)
        self.ageTwoSpectral.fit(self.ageTwoScaled)

        self.ageTwoInverted['spectral'] = self.ageTwoSpectral.labels_

        if writeToFile:
            writeDataFile(self.ageTwoInverted, '01234', 2, self.resultDataRoute)
            writeModels(self.ageTwoKmeans, self.ageTwoMiniBatch, self.ageTwoSpectral, 2, self.modelsRoute)

        # map cluster centers to A-E values
        kmeansDict = getAgeTwoMapping(self.ageTwoKmeans.cluster_centers_)
        self.ageTwoInverted.replace({'kmeans': kmeansDict}, inplace=True)

        miniBatchDict = getAgeTwoMapping(self.ageTwoMiniBatch.cluster_centers_)
        self.ageTwoInverted.replace({'minibatch': miniBatchDict}, inplace=True)

        nearestCentrid = NearestCentroid()
        nearestCentrid.fit(self.ageTwoScaled, self.ageTwoInverted['spectral'])

        spectralDict = getAgeTwoMapping(nearestCentrid.centroids_)
        self.ageTwoInverted.replace({'spectral': spectralDict}, inplace=True)

        if writeToFile:
            writeDataFile(self.ageTwoInverted, 'ABCDE', 2, self.resultDataRoute)

        # create plot with 3 subplot
        createClusterResultsPlot(self.ageTwoInverted, self.CatCols5, 2, writeToFile, graphLang)
        print('-- Cluster info --')
        print(' - A : norm distance - norm speed')
        print(' - B : high distance - norm speed')
        print(' - C : ext. distance - norm/high speed')
        print(' - D : high distance - high speed')
        print(' - E : norm distance - high speed')

    def getTrainingSuggestions(self, ageGroup, clustering, trainType,
                               trainDifficulty):
        if ageGroup == 0:
            data = self.ageNullInverted
        elif ageGroup == 1:
            data = self.ageOneInverted
        else:
            data = self.ageTwoInverted
        plt.rcParams["figure.figsize"] = [8, 8]
        suggestion = getTrainingValue(clustering, trainType, trainDifficulty, data)
        exceptedMovingTime = getExceptedMovingTime(suggestion['average_speed'], suggestion['distance'])
        hourTime, minuteTime = divmod(exceptedMovingTime, 60)

        clusterNames = ['A', 'B', 'C', 'D', 'E']
        plotMarker = 'o'
        for i in range(0, len(clusterNames)):
            plt.plot(data[data[clustering] == clusterNames[i]]['average_speed'],
                     data[data[clustering] == clusterNames[i]]['distance'], plotMarker, c=self.CatCols5[i], label=clusterNames[i])

        plt.plot(suggestion['average_speed'], suggestion['distance'], plotMarker, c='#41cc3d')
        plt.legend(loc='upper left')

        plt.show()

        print('Training suggestion: ')
        print(' - distance: ' + str(suggestion['distance']) + ' m  (' + str(round(suggestion['distance'] / 1000, 2)) + ' km)')
        print(' - average speed: ' + str(suggestion['average_speed']) + ' m/s  (' + str(round(self.mps_to_kmph(suggestion['average_speed']),2)) + ' km/h)')
        print(' - moving time: ' + str(round(hourTime)) + ' hour(s) and ' + str(round(minuteTime)) + ' minute(s)')
        return {'distance': suggestion['distance'], 'average_speed': suggestion['average_speed'], 'moving_time': exceptedMovingTime}

    def mps_to_kmph(self, m_per_s):
        return m_per_s * 3.6

    def kmph_to_mps(self, km_h):
        return km_h / 3.6
## CLASS ENDS HERE
def getExceptedMovingTime(speed, distance, speed_type=''): # moving time based on distance and average speed, return with minutes
    if speed_type == '' or speed_type == 'ms':
        return (distance / 1000) / (speed * 3.6) * 60
    if speed_type == 'kmh':
        return distance / speed * 60
    return 'please use valid speedType (mps or kmph)'

def writeModels(kmeans, minibatch, spectral, ageGroup, route):
    if ageGroup == 0 or ageGroup == 1 or ageGroup == 2:
        print('Writing model pickle files for age group '+str(ageGroup))
        # filename prefix: time_type_data_model.p
        # for example: moving_time_all_ridge.p
        pickle.dump(kmeans, open(route + 'age_' + str(ageGroup) + '_kmeans.p', 'wb'))
        pickle.dump(minibatch, open(route + 'age_' + str(ageGroup) + '_minibatch.p', 'wb'))
        pickle.dump(spectral, open(route + 'age_' + str(ageGroup) + '_spectral.p', 'wb'))
    else:
        print('ERROR: invalid ageGroup in writeModels: '+ str(ageGroup) +'. Use one of the following values: 0, 1, 2')

def writeDataFile(dataset, clusterSuffix, ageGroup, route):
    if ageGroup == 0 or ageGroup == 1 or ageGroup == 2:
        dataset.to_csv(route + 'age_' + str(ageGroup) + '_' + clusterSuffix+'.csv', index=False)
    else:
        print('ERROR: invalid ageGroup in writeDatFile: '+ str(ageGroup) +'. Use one of the following values: 0, 1, 2')

def getAgeNullMapping(centers):
    # map 0-4 cluster IDs to A-E values.
    # A : norm - high distance & norm speed
    # B : high distance & norm - high speed
    # C : ext. distance & high - ext. high speed
    # D : norm distance & high speed
    # E : norm - high distance & ext. high speed
    indices = np.array([[0.0],[1.0],[2.0],[3.0],[4.0]])
    centersWithIndex = np.hstack((centers, indices))
    sortInd = np.lexsort((centersWithIndex[:, 0], centersWithIndex[:, 1]))
    names = [-1,-1,-1,-1,-1]
    if centersWithIndex[sortInd[0]][0] < centersWithIndex[sortInd[1]][0]:# check which average_speed is smaller
        names[sortInd[0]] = 'A'
        names[sortInd[1]] = 'D'
    else:
        names[sortInd[0]] = 'D'
        names[sortInd[1]] = 'A'
    if centersWithIndex[sortInd[2]][0] < centersWithIndex[sortInd[3]][0]:# check which average_speed is smaller
        names[sortInd[2]] = 'B'
        names[sortInd[3]] = 'E'
    else:
        names[sortInd[2]] = 'E'
        names[sortInd[3]] = 'B'
    names[sortInd[4]] = 'C'
    return {0: names[0], 1: names[1], 2: names[2], 3: names[3], 4: names[4]}

def getAgeOneMapping(centers):
    # map 0-4 cluster IDs to A-E values.
    # A - norm to high distance - norm to ext. high speed
    # B - high distance - norm speed
    # C - ext. high distance - norm to ext. high speed
    # D - norm distance - high speed
    # E - norm to high distance - ext. high speed
    indices = np.array([[0.0],[1.0],[2.0],[3.0],[4.0]])
    centersWithIndex = np.hstack((centers, indices))
    sortInd = np.lexsort((centersWithIndex[:, 0], centersWithIndex[:, 1]))
    names = [-1,-1,-1,-1,-1]
    minIndex = np.argmin(centersWithIndex[sortInd[0:3]], axis=0)[0]     # find minimum avg_speed from the three smallest distance
    maxIndex = np.argmax(centersWithIndex[sortInd[0:3]], axis=0)[0]     # find maximum avg_speed from the three smallest distance
    names[sortInd[minIndex]] = 'A'
    names[sortInd[maxIndex]] = 'E'
    names[sortInd[3]] = 'B'
    names[sortInd[4]] = 'C'
    for i in range(0,len(names)):
        if names[i] == -1:
            names[i] = 'D'      # assign D for the last center

    return {0: names[0], 1: names[1], 2: names[2], 3: names[3], 4: names[4]}

def getAgeTwoMapping(centers):
    # map 0-4 cluster IDs to A-E values.
    # A - norm distance - norm speed
    # B - high distance - norm speed
    # C - ext. distance - norm/high speed
    # D - high distance - high speed
    # E - norm distance - high speed
    indices = np.array([[0.0],[1.0],[2.0],[3.0],[4.0]])
    centersWithIndex = np.hstack((centers, indices))
    sortInd = np.lexsort((centersWithIndex[:, 0], centersWithIndex[:, 1]))
    names = [-1,-1,-1,-1,-1]
    if centersWithIndex[sortInd[0]][0] < centersWithIndex[sortInd[1]][0]:# check which average_speed is smaller
        names[sortInd[0]] = 'A'
        names[sortInd[1]] = 'E'
    else:
        names[sortInd[0]] = 'E'
        names[sortInd[1]] = 'A'
    if centersWithIndex[sortInd[2]][0] < centersWithIndex[sortInd[3]][0]:# check which average_speed is smaller
        names[sortInd[2]] = 'B'
        names[sortInd[3]] = 'D'
    else:
        names[sortInd[2]] = 'D'
        names[sortInd[3]] = 'B'
    names[sortInd[4]] = 'C'
    return {0: names[0], 1: names[1], 2: names[2], 3: names[3], 4: names[4]}

def getTrainingValue(clustering, trainType, trainDifficulty, dataset):
    return dataset[dataset[clustering] ==
                   trainType].quantile(trainDifficulty)

def plotClusterResult(dataset, colors, ageGroup, clustering, labels, writeToFile=False, language='en'):
    c_column = dataset[clustering]
    clusterNames = ['A', 'B', 'C', 'D', 'E']
    lines = [0, 1, 2, 3, 4]
    plotMarker = 'o'
    fontSize = 16

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    # fig.suptitle(graphTitle)

    for i in range(0, len(clusterNames)):
        lines[i] = ax.plot(dataset[c_column == clusterNames[i]]['average_speed'],
                           dataset[c_column == clusterNames[i]]['distance'], plotMarker, c=colors[i], label=clusterNames[i])
    ax.set_xlabel(labels['xLabel'], fontsize=fontSize)
    ax.set_ylabel(labels['yLabel'], fontsize=fontSize)
    ax.set_title(labels[clustering], fontsize=fontSize)
    ax.tick_params(axis='both', labelsize=fontSize)
    fig.legend(lines[0] + lines[1] + lines[2] + lines[3] + lines[4], clusterNames, loc=(0.24, 0.72))# from left, from bottom, %
    plt.tight_layout()
    if writeToFile:
        plt.savefig(fname='./graphs/age_group_' + str(ageGroup) + '_' + str(clustering)+'_results.png', dpi=1000)
    plt.show()

def createClusterResultsPlot(dataset, colors, ageGroup, writeToFile=False, language='en'):

    if language == 'en':
        labels = {'graphTitle' : 'Comparing clustering resuls',
                'kmeans': 'K-Means',
                'minibatch': 'Mini Batch K-Means',
                'spectral': 'Spectral',
                'xLabel': 'Average speed (m/s)',
                'yLabel': 'Distance (m)'}
    else:
        labels = {'graphTitle': 'Klaszterezési eredmények összehasonlítása',
                  'kmeans': 'K-Means',
                  'minibatch': 'Mini Batch K-Means',
                  'spectral': 'Spectral',
                  'xLabel': 'Átlagos sebesség (m/s)',
                  'yLabel': 'Távolság (m)'}

    plotClusterResult(dataset, colors, ageGroup, 'kmeans', labels, writeToFile, language)
    plotClusterResult(dataset, colors, ageGroup, 'minibatch', labels, writeToFile, language)
    plotClusterResult(dataset, colors, ageGroup, 'spectral', labels, writeToFile, language)
