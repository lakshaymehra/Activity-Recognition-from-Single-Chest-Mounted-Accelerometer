# Run using:  python .\exploratory_data_analysis.py --directory_path 'C:\Users\mehra\PycharmProjects\VicaraAssignment\Activity Recognition from Single Chest-Mounted Accelerometer\Activity Recognition from Single Chest-Mounted Accelerometer'

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import argparse

def plot_activity_counts(df):

    activity_counts = df['Activity_ID'].value_counts()
    # print(activity_counts)
    activity_counts.plot(kind='bar', title="Activities")
    plt.show()

def plot_crosstab(df):

    test = pd.crosstab(index=df['User_ID'], columns=df['Activity_ID'])
    test.plot(kind="barh", stacked=True, figsize=(20, 5),title = 'Cross-Tab')
    plt.show()

def plot_acceleration_trends_for_each_activity(df):

    labels = ['No Activity', 'Working at Computer', 'Standing Up, Walking and Going up\down stairs', 'Standing',
              'Walking', 'Going Up\Down Stairs', 'Walking and Talking with Someone', 'Talking while Standing']

    for i in range(0,8):
        sub = df[df['Activity_ID'] == i]
        sub = sub[['X-acceleration', 'Y-acceleration', 'Z-acceleration']]
        sub = sub[:40000]
        sub.plot(subplots=True, figsize=(15, 5),title=labels[i])
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory_path',
        type=str,
        help='Specify the path to the directory containing the data.')

    args = parser.parse_args()
    directory_path = args.directory_path

    dir = glob(directory_path + "/*.csv")
    df = pd.DataFrame()

    for i, name in enumerate(dir):
        temp_df = pd.read_csv(name, header=None)
        temp_df['User_ID'] = i + 1
        df = df.append(temp_df)

    del df[0]

    df.columns = ['X-acceleration', 'Y-acceleration', 'Z-acceleration', 'Activity_ID', 'User_ID']
    print(df.head())

    plot_activity_counts(df)
    plot_crosstab(df)
    plot_acceleration_trends_for_each_activity(df)

