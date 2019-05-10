import pandas
import pandas_profiling
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings



# module notes:
# put inside the class the callable funtions
# put everything outside of the class you do not want to be callable from the notebook

class DataProcessor():
    def __init__(self, dataRoute, cleanedRoute, resultsRoute, profileRoute, rawFileName, describeFileName, cleanedDescribeFileName,dateParser = lambda dates: pandas.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')):   # define data routes and csv names!
        self.dataRoute = dataRoute
        self.cleanedRoute = cleanedRoute
        self.resultsRoute = resultsRoute
        self.profileRoute = profileRoute
        self.rawFileName = rawFileName
        self.describeFileName = describeFileName
        self.cleanedDescribeFileName = cleanedDescribeFileName
        self.dateParser = dateParser
        self.rawData = None
        self.cleanedData = None
        self.graphData = None
        self.uselessColumns = ['heartrate_opt_out', 'average_watts', 'commute', 'device_watts', 'flagged', 'has_heartrate',
                               'kilojoules', 'max_watts', 'pr_count', 'resource_state', 'weighted_average_watts', 'sex',
                               'start_latlng', 'end_latlng']
        self.CatCols8 = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
        self.CatCols5 = ['#003f5c', '#3d61f4', '#bc5090', '#ff6361', '#ffa600']
        self.CatCols3 = ['#003f5c', '#bc5090', '#ffa600']

    def readRawData(self, generate_description=False):
        self.rawData = pandas.read_csv(self.dataRoute+self.rawFileName, encoding='UTF-8', sep='|', parse_dates=['start_date_local'], date_parser=self.dateParser)
        self.cleanedData = self.rawData.copy()
        for col in self.cleanedData.columns:
            nanValues = self.cleanedData[col].isnull()
            column_name = col
            self.cleanedData.loc[nanValues, column_name] = None
        print('You can access the raw data under: .rawData')
        print('You can access the cleaning data under: .cleanedData')
        print('Data sample:')
        print(self.rawData.head())
        if generate_description:
            profileData(self.cleanedData, self.profileRoute, 'rawDataProfiling.html')
            rawDesc = self.cleanedData.describe()
            rawDesc.drop(columns='hashed_id')
            rawDesc = rawDesc.T
            rawDesc.columns = ['count', 'mean', 'std', 'min', '0.25', '0.50', '0.75', 'max']
            rawDesc = round(rawDesc, 2)
            rawDesc = rawDesc.astype({'count': int})
            describeData(self.cleanedData, self.profileRoute, 'rawDataDescription.csv')
            print('Pandas profiling writen to: '+self.profileRoute+'rawDataProfiling.html')
            print('Description of numeric columns writen to: '+self.dataRoute+'rawDataDescription.csv')
        print(' ')

        print('Calculating useful and useless columns')
        for col in self.cleanedData.columns:
            useless = self.cleanedData[col].isnull().sum()
            #isUseless = (useless == len(self.cleanedData))
            if useless == len(self.cleanedData):
                self.uselessColumns.append(col)
                print(" - " + str(col) + " has only None and False values")
            else:
                print(" - " + str(col) + " has useful values - " + str(
                    '%.2f' % (100 * (len(self.cleanedData) - useless) / len(self.cleanedData))) + "%")
        print(' ')

        print('useless columns: ')
        for c in self.uselessColumns:
            print(' - '+str(c))


    def dropUselesColumns(self):
        self.cleanedData.drop(columns=self.uselessColumns, inplace=True)
        print('useless columns dropped')

    def detectAndDropOutliers(self):
        print('--- average_speed ---')
        averageSpeedIndex = getOutliersForFeature(self.cleanedData, 'average_speed')
        self.cleanedData.drop(self.cleanedData.index[averageSpeedIndex], inplace=True)
        self.cleanedData.reset_index(drop=True, inplace=True)
        print('data length:'+str(len(self.cleanedData)))
        print('')

        print('--- distance ---')
        distanceIndex = getOutliersForFeature(self.cleanedData, 'distance')
        print('the outlying values are between 114176 and 436806 meters (114 km and 436 km). These are valid values, '
              'should be treated as a standalone group. These values will not be dropped in this section.')
        print('data length:'+str(len(self.cleanedData)))
        print('')

        print('--- elev_high ---')
        elevHighIndex = getOutliersForFeature(self.cleanedData, 'elev_high')
        self.cleanedData.drop(self.cleanedData.index[elevHighIndex], inplace=True)
        self.cleanedData.reset_index(drop=True, inplace=True)
        print('data length:'+str(len(self.cleanedData)))
        print('')

        print('--- max_speed ---')
        maxSpeedIndex = getOutliersForFeature(self.cleanedData, 'max_speed')
        self.cleanedData.drop(self.cleanedData.index[maxSpeedIndex], inplace=True)
        self.cleanedData.reset_index(drop=True, inplace=True)
        print('data length:'+str(len(self.cleanedData)))
        print('')

        print('--- moving_time ---')
        movingTimeIndex = getOutliersForFeature(self.cleanedData, 'moving_time')
        print('The moving_time is highly correlated with the distance, the high values are belong to the outlying '
              'distance values. These will be kept now for future usability.')
        print('data length:'+str(len(self.cleanedData)))
        print('')

        print('There are still some invalid values. To remove these I defined some static rules. Every row will be '
              'dropped where:')
        print('distance <= 100 meter')
        print('average_speed <= 2 m/s')
        print('workout_type == 0 or workout_type == 4  <- these are invalid values')
        print('')

        self.cleanedData = self.cleanedData[self.cleanedData['average_speed'] > 2]
        self.cleanedData = self.cleanedData[self.cleanedData['distance'] > 100]
        self.cleanedData = self.cleanedData[(self.cleanedData['workout_type'] != 0) & (self.cleanedData['workout_type'] != 4)]
        self.cleanedData.reset_index(drop=True, inplace=True)
        # elapsed_time should not be less than moving_time -> change elapsed_time value where it is less
        indices = self.cleanedData[self.cleanedData['elapsed_time'] < self.cleanedData['moving_time']].index
        for i in indices:
            self.cleanedData['elapsed_time'][i] = self.cleanedData['moving_time'][i].copy()

        print('There are some null values: ')
        print(len(self.cleanedData) - self.cleanedData.count())
        self.cleanedData = self.cleanedData[self.cleanedData['elev_high'].notnull()]
        self.cleanedData = self.cleanedData[self.cleanedData['elev_low'].notnull()]
        self.cleanedData.reset_index(drop=True, inplace=True)
        print('The elev_high and elev_low values can not be fixed so they are dropped. The trainer and the workout_type'
              ' columns will be one-hot encoded later.')
        print('data length:'+str(len(self.cleanedData)))
        self.graphData = self.cleanedData.copy()
        print('')
        print('The data is saved to the graphData variable for easier visualization (the cleanedData will change later)')



    def showWeeklyInsightGraph(self, language='en', writeToImg=False):
        # prepare data for graph
        data = self.graphData[['age_group', 'distance', 'average_speed', 'start_date_local']].copy()
        data = transformStartDateToDays(data)
        ageVisual = data[['age_group', 'day_of_week', 'distance']].groupby(['day_of_week', 'age_group'])[
            'distance'].agg(['count', 'mean']).reset_index().rename(
            columns={'count': 'no_of_activities', 'mean': 'average_distance'})

        if language == 'en':
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            mapping = {day: i for i, day in enumerate(days)}
            ageLabels = ['Age group 0 (18-30)', 'Age group 1 (31-45)', 'Age group 2 (46+)']
            weekDayLabel = 'Day of week'
            frequencyLabel = 'Frequency (%)'
            titleLabel1 = 'Average distance'
            titleLabel2 = 'Activity freqency'
            distanceLabel = 'Distance (km)'
        else:
            days =['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']
            mapping = {day: i for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}
            ageLabels = ['0. korcsoport (18-30)', '1. korcsoport (31-45)', '2. korcsoport (46+)']
            weekDayLabel = 'Hét napjai'
            frequencyLabel = 'Gyakoriság (%)'
            titleLabel1 = 'Átlagos távolság'
            titleLabel2 = 'Útvonal gyakoriság'
            distanceLabel = 'Távolság (km)'

        key = ageVisual['day_of_week'].map(mapping)
        ageVisual = ageVisual.iloc[key.argsort()].set_index('day_of_week')

        # create figure, set default values
        fig, (ax2, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
        index = [0, 1, 2, 3, 4, 5, 6]
        bar_width = 0.2
        opacity = 0.8
        age = ageVisual['age_group']
        # create distance plot - first subplot
        # ageNullSum = sum(ageVisual[age == 0]['average_distance'])
        # 0. age group
        l1 = ax1.bar(index, ageVisual[age == 0]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[0],
                     label=ageLabels[0])

        # ageOneSum = sum(ageVisual[age == 1]['average_distance'])
        # 1. age group
        l2 = ax1.bar([x + bar_width for x in index], ageVisual[age == 1]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[1],
                     label=ageLabels[1])

        # ageTwoSum = sum(ageVisual[age == 2]['average_distance'])
        # 2. age group
        l3 = ax1.bar([x + 2 * bar_width for x in index], ageVisual[age == 2]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[2],
                     label=ageLabels[2])
        # set info on first subplot
        ax1.set_xlabel(weekDayLabel)
        ax1.set_ylabel(distanceLabel)
        ax1.set_title(titleLabel1)
        ax1.set_xticks([x + bar_width for x in index])
        ax1.set_xticklabels(days)
        # ax1.setlegend(loc=2)

        # create frequency plot - second subplot
        # 0. age group
        ageNullSum = sum(ageVisual[age == 0]['no_of_activities'])
        ax2.bar(index, ageVisual[age == 0]['no_of_activities'] / ageNullSum * 100, bar_width,
                alpha=opacity,
                color=self.CatCols3[0],
                label=ageLabels[0])
        # 1. age group
        ageOneSum = sum(ageVisual[age == 1]['no_of_activities'])
        ax2.bar([x + bar_width for x in index], ageVisual[age == 1]['no_of_activities'] / ageOneSum * 100, bar_width,
                alpha=opacity,
                color=self.CatCols3[1],
                label=ageLabels[1])
        # 2. age group
        ageTwoSum = sum(ageVisual[age == 2]['no_of_activities'])
        ax2.bar([x + 2 * bar_width for x in index], ageVisual[age == 2]['no_of_activities'] / ageTwoSum * 100,
                bar_width,
                alpha=opacity,
                color=self.CatCols3[2],
                label=ageLabels[2])
        # set info on second subplot
        ax2.set_xlabel(weekDayLabel)
        ax2.set_ylabel(frequencyLabel)
        ax2.set_title(titleLabel2)
        ax2.set_xticks([x + bar_width for x in index])
        ax2.set_xticklabels(days)

        # set global legend
        fig.legend([l1, l2, l3], ['0. korcsoport (18-30)', '1. korcsoport (31-45)', '2. korcsoport (46 +)'],
                   loc=(0.08, 0.77))  # from left, from bottom, percentage

        plt.setp(ax1.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        plt.tight_layout()

        if writeToImg:
            plt.savefig(fname='./graphs/FrequencyAndDistanceOnWeekDays.png', dpi=1000)
            print('The figure is saved to: ./graphs/FrequencyAndDistanceOnWeekDays.png')
        else:
            print('If you want to save the figure as a png set the writeToFile params to True')

        plt.show()

    def showDailyInsightGraph(self, language='en', writeToImg=False):
        # prepare data for graph
        data = self.graphData[['age_group', 'distance', 'average_speed', 'start_date_local']].copy()
        data = transformStartDateToDays(data)

        ageVisual = data[['age_group', 'daypart', 'distance']].groupby(['daypart', 'age_group'])['distance'].agg(
            ['count', 'mean']).reset_index().rename(columns={'count': 'no_of_activities', 'mean': 'average_distance'})

        if language == 'en': # english labels
            dayParts = ['dawn', 'morning', 'forenoon', 'afternoon', 'evening', 'night']
            mapping = {day: i for i, day in enumerate(dayParts)}
            ageLabels = ['Age group 0 (18-30)', 'Age group 1 (31-45)', 'Age group 2 (46+)']
            dayPartLabel = 'Dayparts'
            frequencyLabel = 'Frequency (%)'
            titleLabel1 = 'Average distance'
            titleLabel2 = 'Activity freqency'
            distanceLabel = 'Distance (km)'
        else:                # hungarian labels
            dayParts = ['hajnal', 'reggel', 'délelőtt', 'délután', 'este', 'éjszaka']
            mapping = {day: i for i, day in enumerate(['dawn', 'morning', 'forenoon', 'afternoon', 'evening', 'night'])}
            ageLabels = ['0. korcsoport (18-30)', '1. korcsoport (31-45)', '2. korcsoport (46+)']
            dayPartLabel = 'Napszakok'
            frequencyLabel = 'Gyakoriság (%)'
            titleLabel1 = 'Átlagos távolság'
            titleLabel2 = 'Útvonal gyakoriság'
            distanceLabel = 'Távolság (km)'

        key = ageVisual['daypart'].map(mapping)

        ageVisual = ageVisual.iloc[key.argsort()].set_index('daypart')

        # create figure, set default values
        fig, (ax2, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
        index = [0, 1, 2, 3, 4, 5]
        bar_width = 0.2
        opacity = 0.8
        age = ageVisual['age_group']

        # create distance plot - first subplot
        # 0. age group
        # ageNullSum = sum(ageVisual[age == 0]['average_distance'])
        l1 = ax1.bar(index, ageVisual[age == 0]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[0],
                     label=ageLabels[0])
        # 1. age group
        # ageOneSum = sum(ageVisual[age == 1]['average_distance'])
        l2 = ax1.bar([x + bar_width for x in index], ageVisual[age == 1]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[1],
                     label=ageLabels[1])
        # 2. age group
        # ageTwoSum = sum(ageVisual[age == 2]['average_distance'])
        l3 = ax1.bar([x + 2 * bar_width for x in index], ageVisual[age == 2]['average_distance'] / 1000, bar_width,
                     alpha=opacity,
                     color=self.CatCols3[2],
                     label=ageLabels[2])
        # set info on first subplot
        ax1.set_xlabel(dayPartLabel)
        ax1.set_ylabel(distanceLabel)
        ax1.set_title(titleLabel1)
        ax1.set_xticks([x + bar_width for x in index])
        ax1.set_xticklabels(dayParts)

        # create frequency plot - second subplot
        # 0. age group
        ageNullSum = sum(ageVisual[age == 0]['no_of_activities'])
        ax2.bar(index, ageVisual[age == 0]['no_of_activities'] / ageNullSum * 100, bar_width,
                alpha=opacity,
                color=self.CatCols3[0],
                label=ageLabels[0])
        # 1. age group
        ageOneSum = sum(ageVisual[age == 1]['no_of_activities'])
        ax2.bar([x + bar_width for x in index], ageVisual[age == 1]['no_of_activities'] / ageOneSum * 100, bar_width,
                alpha=opacity,
                color=self.CatCols3[1],
                label=ageLabels[1])
        # 2. age group
        ageTwoSum = sum(ageVisual[age == 2]['no_of_activities'])
        ax2.bar([x + 2 * bar_width for x in index], ageVisual[age == 2]['no_of_activities'] / ageTwoSum * 100,
                bar_width,
                alpha=opacity,
                color=self.CatCols3[2],
                label=ageLabels[2])
        # set info on second subplot
        ax2.set_xlabel(dayPartLabel)
        ax2.set_ylabel(frequencyLabel)
        ax2.set_title(titleLabel2)
        ax2.set_xticks([x + bar_width for x in index])
        ax2.set_xticklabels(dayParts)

        # set global legend
        fig.legend([l1, l2, l3], ageLabels,
                   loc=(0.08, 0.77))

        plt.setp(ax1.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        plt.tight_layout()

        if writeToImg:
            plt.savefig(fname='./graphs/FrequencyAndDistanceOnDayparts.png', dpi=1000)
            print('The figure is saved to: ./graphs/FrequencyAndDistanceOnDayparts.png')
        else:
            print('If you want to save the figure as a png set the writeToFile params to True')

        plt.show()

    def oneHotEncodeFeatures(self):
        print('--- age_group ---')
        ageOneHot = pandas.get_dummies(self.cleanedData['age_group'], prefix='age')
        self.cleanedData.drop(columns=['age_group'], inplace=True)
        self.cleanedData = self.cleanedData.join(ageOneHot)
        print('new columns: ')
        print(' - age_0.0')
        print(' - age_1.0')
        print(' - age_2.0')
        print('')

        print('--- trainer ---')
        self.cleanedData['trainer_onehot'] = self.cleanedData.apply(lambda row: transformTrainerColumn(row), axis=1)
        self.cleanedData.drop(columns=['trainer'], inplace=True)
        print('new column: ')
        print(' - trainer_onehot')
        print('')

        print('--- workout_type ---')
        workoutTypeOneHot = pandas.get_dummies(self.cleanedData['workout_type'], prefix='workout_type')
        self.cleanedData.drop(columns='workout_type', inplace=True)
        self.cleanedData = self.cleanedData.join(workoutTypeOneHot)
        print('new columns: ')
        print(' - workout_type_10.0')
        print(' - workout_type_11.0')
        print(' - workout_type_12.0')
        print('')

        print('--- start_date_local ---')
        print('this feature will be converted to weekdays and dayparts')
        self.cleanedData = transformStartDateAndOneHot(self.cleanedData)
        print('new columns: ')
        print(' - weekday_Monday')
        print(' - weekday_Tuesday')
        print(' - weekday_Wednesday')
        print(' - weekday_Thursday')
        print(' - weekday_Friday')
        print(' - weekday_Saturday')
        print(' - weekday_Sunday')
        print(' - daypart_dawn')
        print(' - daypart_morning')
        print(' - daypart_forenoon')
        print(' - daypart_afternoon')
        print(' - daypart_evening')
        print(' - daypart_night')
        print('')

        self.cleanedData.reset_index(drop=True, inplace=True)
        print('data length: ' + str(len(self.cleanedData)))
        print('call .cleanedData.head() to view the first 5 row of the dataset')

    def saveAndProfileCleanedData(self):
        # TODO: create description file
        profileData(self.cleanedData, self.dataRoute,'profiled_cleaned_2019_03_20.html')
        self.cleanedData.to_csv(self.cleanedRoute + "cleaned_" + self.rawFileName, sep="|", index=False)
        print('pandas profiling written to: ' + self.dataRoute + 'profiled_cleaned_2019_03_20.html')
        print('cleaned dataset written to: ' + self.cleanedRoute + 'cleaned_' + self.rawFileName)


    def getExceptedMovingTime(self, speed, distance, speed_type=''):
        if speed_type == '' or speed_type == 'ms':
            return (distance / 1000) / (speed * 3.6)
        if speed_type == 'kmh':
            return distance / speed
        return 'please use valid speedType (mps or kmph)'

    def mps_to_kmph(self, m_per_s):
        return m_per_s * 3.6

    def kmph_to_mps(self, km_h):
        return km_h / 3.6

# CLASS ENDS HERE #
def transformStartDateToDays(dataset):
    dataset['day_of_week'] = dataset['start_date_local'].dt.day_name()
    dataset['daypart'] = dataset['start_date_local'].apply(lambda row: timeToPartOfDay(row))
    return dataset


def profileData(dataset, profiledRoute, filename):
    profiled = pandas_profiling.ProfileReport(dataset)
    profiled.to_file(outputfile=profiledRoute+filename)

def describeData(dataset, route, filename):
    dataDescription = dataset.describe().T
    dataDescription.to_csv(route+filename+'_DESCRIPTION.csv')

def transformTrainerColumn(row):
    if row['trainer'] == True:
        return 1
    return 0


def timeToPartOfDay(time):
    # defive subjective dayparts based on time
    if time.hour >= 3 and time.hour < 6:
        return 'dawn'
    if time.hour >= 6 and time.hour < 9:
        return 'morning'
    if time.hour >= 9 and time.hour < 12:
        return 'forenoon'
    if time.hour >= 12 and time.hour < 18:
        return 'afternoon'
    if time.hour >= 18 and time.hour < 22:
        return 'evening'
    return 'night'


def transformStartDateAndOneHot(dataset):
    # split the original date column to weekdays and dayparts colummns (and apply one-hot encoding)
    dataset['day_of_week'] = dataset['start_date_local'].dt.day_name()
    # [d.weekday() for d in dataset['start_date_local']] # return with an integer, Monday = 0
    weekDayOneHot = pandas.get_dummies(dataset['day_of_week'], prefix='weekday')
    dataset.drop(columns='day_of_week', inplace=True)
    dataset = dataset.join(weekDayOneHot)

    dataset['daypart'] = dataset['start_date_local'].apply(lambda row: timeToPartOfDay(row))
    partOfDayOneHot = pandas.get_dummies(dataset['daypart'], prefix='daypart')
    dataset.drop(columns='daypart', inplace=True)
    dataset = dataset.join(partOfDayOneHot)
    dataset.drop(columns='start_date_local', inplace=True)
    return dataset


def getOutliersForFeature(data, feature):
    mean = data[feature].mean()
    variance = math.sqrt(data[feature].var())
    indeces = []
    for index, row in data.iterrows():
        if abs((row[feature] - mean) / variance) > 2.5:
            indeces.append(index)
    print("number of outliers: " + str(len(indeces)))

    return indeces






