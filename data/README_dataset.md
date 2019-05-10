# Datasets

In this folder you can find all the different data source for the project and the cleaned, modified datasets.

## JSON

There are a bunch of .json file. Their names are indicating the dates of exports from DB and the version. For example:
```
2019_03_05_14h_v2.json
```
was exported from the DB on 5 March, 2019 afternoon (14 h) and was modified by hand (-1 and values to NaN-s) 
## CSV

There are 2 general names for the .csv files:
```
data_yyyy_mm_dd.csv
```
and 
```
data_yyyy_mm_dd_DESCRIPTION.csv
```
The first one is containing the useful values for Machine Learning, generated from the corresponding .json file. This is easy-to-use format for python, does not contains sensitive data but it is "raw", without any cleaning or onehot encoding. 

The second type of csv file is containing the description of the features from the corresponding csv file. It is generated with the following command:
```
import pandas

dataDescription = rawData.describe().T
dataDescription.to_csv(dataRoute+csvFileName+'_DESCRIPTION.csv')
```
## Cleaned folder

The cleaned folder contains the cleaned datasets, generated from the raw data.
##### dropped_data_2019_03_05.csv
Contains the following features:
* age_group
* average_speed
* average_watts
* commute
* device_watts
* distance
* elapsed_time
* elev_high
* elev_low
* end_latlng
* flagged
* has_heartrate
* hashed_id
* kilojoules
* max_speed
* max_watts
* moving_time
* pr_count
* resource_state
* sex
* start_date_local
* start_latlng
* total_elevation_gain
* trainer
* weighted_average_watts
* workout_type

and created by dropping rows from the original row data where the following columns contained 0.0 values: **average_speed**, **distance**, **elapsed_time**, **max_speed**.
This is a very rudimentary first approach and should not be used for later ML development.

#### advanced_2019_03_05.csv
Result of a more advanced data cleaning process, contains onehot encoded features. 
Features:
* age_group
* average_speed
* distance
* elapsed_time
* elev_high
* elev_low
* hashed_id
* max_speed
* moving_time
* sex
* total_elevation_gain
* trainer_onehot
* workout_type_0.0
* workout_type_4.0
* workout_type_10.0
* workout_type_11.0
* workout_type_12.0
* weekday_Friday
* weekday_Monday
* weekday_Saturday
* weekday_Sunday
* weekday_Thursday
* weekday_Tuesday
* weekday_Wednesday
* daypart_afternoon
* daypart_dawn
* daypart_evening
* daypart_forenoon
* daypart_morning
* daypart_night

Problems:
* **elev_high**: missing values (167)
* **elev_low**: missing values (168)
* **distance**: 0 values (50) 
* **average_speed**: 0 values (50)
* **elapsed_time**: 0 values (5) - there are moving_time for 2 of them, distance for 3 of them and average_speed for 2 of them 
* **max_speed**: 0 values (124)