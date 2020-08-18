import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
import csv
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# #################################################### Functions #######################################################

# These functions are mainly used so as not to load massive data for every single function. So by uncommenting each
# piece of code, one task is implemented and its desired result is showed.


def data_loader(file_name, usecolumns, skiprows):
    output_data_frame = pd.read_csv(file_name, usecols=usecolumns, skiprows=skiprows)
    return output_data_frame


def search_result(user_id, file_name):
    global traindf

    # For this task we need to load 2 data files one of which is loaded by running the main (traindf)

    usersdf = data_loader(file_name, [0, 1, 2, 3, 4, 5, 6, 7, 8], skiprows=0)

    # user's info in train.csv dataframe
    user_info_train_index = (traindf['user_id'] == user_id)

    # user's info in users.csv dataframe
    user_info_users_index = (usersdf['user_id'] == user_id)

    user_info_train = traindf[user_info_train_index]
    user_info_users = usersdf[user_info_users_index]

    # writing the search result to a text file
    with open('search_result.txt', 'w') as f:
        user_info_train.to_csv(f, sep=',', index=None, header=False)
        user_info_users.to_csv(f, sep=',', index=None, header=False)


def statistics_calculator():
    global traindf

    # mode_delivery_min_by_user = traindf.groupby('user_id', as_index=False)['delivery_min'].value_counts().idxmax()

    # mode_delivery_hour_by_user = traindf.groupby('user_id', as_index=False)['delivery_hour'].value_counts().idxmax()

    # mode_interaction_dow_by_user = traindf.groupby('user_id', as_index=False)['interaction_dow'].value_counts(
    # ).idxmax()

    mean_by_user_id = traindf.groupby('user_id', as_index=False).mean()

    median_by_user_id = traindf.groupby('user_id', as_index=False).median()

    variance_by_user_id = traindf.groupby('user_id', as_index=False).var()

    # mode_by_user = [mode_delivery_min_by_user, mode_delivery_hour_by_user, mode_interaction_dow_by_user]

    return mean_by_user_id, median_by_user_id, variance_by_user_id


def print_statistics():
    global traindf

    # ### Hour

    delivery_hours = traindf.delivery_hour
    delivery_hour_mean = delivery_hours.mean(0)
    delivery_hour_mode = delivery_hours.mode()
    delivery_hour_median = delivery_hours.median(0)
    delivery_hour_var = delivery_hours.var(0)

    print('The average hour a notification arrives is:', delivery_hour_mean)
    print('The mode of hour a notification arrives is:', delivery_hour_mode[0])
    print('The median of hour a notification arrives is:', delivery_hour_median)
    print('The variance of hour a notification arrives is:', delivery_hour_var)

    # ### Minutes

    delivery_minutes = traindf.delivery_min
    delivery_minutes_mean = delivery_minutes.mean(0)
    delivery_minutes_mode = delivery_minutes.mode()
    delivery_minutes_median = delivery_minutes.median(0)
    delivery_minutes_var = delivery_minutes.var(0)

    print('The average minute of minute a notification arrives is:', delivery_minutes_mean)
    print('The mode of minute a notification arrives is:', delivery_minutes_mode[0])
    print('The median of minute a notification arrives is:', delivery_minutes_median)
    print('The variance of minute a notification arrives is:', delivery_minutes_var)

    # ### Days

    delivery_days = traindf.interaction_dow
    delivery_days_mean = delivery_days.mean(0)
    delivery_days_mode = delivery_days.mode()
    delivery_days_median = delivery_days.median(0)
    delivery_days_var = delivery_days.var(0)

    print('The average day a notification has arrived is:', delivery_days_mean)
    print('The mode of day a notification arrives is:', delivery_days_mode[0])
    print('The median of day a notification arrives is:', delivery_days_median)
    print('The variance of day a notification arrives is:', delivery_days_var)


def scatter_plotter():
    # We just need columns 7 and 8 (N1 and N2)

    usersdf = data_loader('users.csv', [7, 8], skiprows=0)

    usersdf.plot.scatter(x='N1', y='N2')
    plt.show()


def box_plotter():
    # We just need columns 9 (N3)

    usersdf = data_loader('users.csv', [9], skiprows=0)

    usersdf.plot.box()
    plt.show()


def distribution_plotter():

    notifs_df = pd.read_csv('notifs.txt', sep="\t", names=['A'])

    notifs_df_splitted = notifs_df.A.str.split(' ', n=43, expand=True)

    words_count = []

    for i in range(max(notifs_df_splitted.shape)):
        for j in range(5, 43):
            if type(notifs_df_splitted.ix[i, j]) is type(None):
                continue
            else:
                temp = int(notifs_df_splitted.ix[i, j])
                words_count.append(temp)

    words_count_series = pd.Series(words_count)

    words_count_series.plot.kde()

    plt.show()


def bar_plotter():
    global traindf

    delivery_mins = traindf['delivery_min'].value_counts(sort=False)
    delivery_mins.plot.bar()
    plt.show()


def pie_plotter():

    notifs_df = pd.read_csv('notifs.txt', sep="\t", names=['A'])

    notifs_df_splitted = notifs_df.A.str.split(' ', n=6, expand=True)

    notif_feature = notifs_df_splitted[4]
    notif_feature_numeric = pd.to_numeric(notif_feature)
    feature_counted = notif_feature_numeric.value_counts()

    feature_counted.plot.pie(autopct='%.2f', figsize=(10, 10))
    plt.show()


def process_na():

    usersdf = data_loader('users.csv', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], skiprows=0)

    nan_percentage = 1 - usersdf.count(0)/len(usersdf.index)

    ...

    usersdf_missing_data_removed = usersdf.drop(usersdf.columns[nan_percentage > 0.15], 1)

    # Now we are going to remove those rows with nan in one of their columns

    usersdf_missing_data_removed.dropna(axis=0, how='any')

    return nan_percentage, usersdf_missing_data_removed


def information_leakage(train_df):

    data_frame_information_leaky_columns_removed = train_df.drop(train_df.columns[[3, 4, 5]], axis=1)

    return data_frame_information_leaky_columns_removed


# ###################################################### Main #########################################################


a = [0, 1, 2, 3, 4, 5, 7, 8]

traindf = data_loader('train.csv', a, skiprows=0)

# user id of the first user
user_id = traindf.loc[0][0]

# ###################################################### Part 3 ######################################################

# #####################  SECTION 3.2: To get the search-result.txt, please uncomment the following code

# search_result(user_id, 'users.csv')

# #####################  SECTION 3.4: To process the N/A values, please uncomment the following code

# nan_percentages, users_dataframe_missing_value_removed = process_na()

# #####################  SECTION 3.5: To remove the information leaky columns, please uncomment the following code

# traindf_information_leaky_columns_removed = information_leakage(traindf)

# ###################################################### Part 4 ######################################################

# #####################  SECTION 4.1:

# # for statistics of delivery days and minutes and hours to be printed, please uncomment the following code

# print_statistics()

# # To get the results for important statistics, please uncomment the following code

# [mean_df, median_df, variance_df] = statistics_calculator()

# #####################  SECTION 4.2: To see statistical graphs, please uncomment the following code

# # Scatter plot of N1 & N2

# scatter_plotter()

# # Box plot for N3

# box_plotter()

# # Distribution plot for words (just the first word of every notification)

# distribution_plotter()

# # Bar plot for delivery minutes

# bar_plotter()

# # Pie plot for word feature

# pie_plotter()

# ###################################################### Part 5 ######################################################

# ################################################# First Method #################################################

# To run this method, please uncomment to where the second method starts.

# testdf = data_loader('test.csv', [0, 1], skiprows=0)
#
# mean_by_user_id = traindf.groupby('user_id', as_index=False).mean()
#
# interactions = mean_by_user_id.interaction
#
# user_id_container = mean_by_user_id.user_id
#
# prediction = []
# for i in tqdm(range(max(testdf.shape))):
#     test_sample_id = testdf.ix[i, 0]
#
#     Truth_value = user_id_container.isin([test_sample_id])
#
#     if Truth_value.any():
#
#         test_sample_id_index = user_id_container[user_id_container == test_sample_id].index[0]
#
#         # print(test_sample_id_index)
#
#         predict = interactions.ix[test_sample_id_index]
#
#     else:
#         predict = 0
#
#     prediction.append(predict)
#
# prediction = pd.Series(prediction)
#
# prediction.to_csv('pp.csv')
#
# temp_prediction = data_loader('pp.csv', [0, 1], skiprows=0)
#
# temp = temp_prediction.xi[:, 1]
#
# temp.values[temp <= 0.1] = 0
# temp.values[temp >= 0.1] = 1
#
# temp_int = temp.astype(int)
#
# testdf['interaction'] = temp_int
#
# testdf.to_csv('output.csv', index=False)

# To run first method, stop uncommenting here!

# ################################################# Second Method #################################################

# ################################################# This code won't work and is to show that I tried this method but
# I could not get satisfying results by that ################################################

# One-hot encoding

# usersdf = data_loader('users.csv', usecolumns=[0, 1, 2, 3, 4, 5, 6, 7, 8], skiprows=0)
#
# user_ids = usersdf.user_id
#
#
# # encoder.fit(usersdf.C3)
#
# notifs_df = pd.read_csv('notifs.txt', sep="\t", names=['A'])
#
# notifs_df_splitted = notifs_df.A.str.split(' ', n=6, expand=True)
#
# notif_feature = notifs_df_splitted[4]
# notif_feature_numeric = pd.to_numeric(notif_feature)
# notif_encoded = pd.get_dummies(notif_feature_numeric)
#
# # X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
# #                                                     train['Survived'], test_size=0.30,
# #                                                     random_state=101)
#
# # x_train = notif_encoded
#
# train_data_not_encoded = traindf.interaction_min
#
# x_train = pd.get_dummies(train_data_not_encoded)
#
# y_train = traindf.interaction
#
# logmodel = LogisticRegression()
# logmodel.fit(x_train, y_train)
# predictions = logmodel.predict(X_test)
############################################################################################
