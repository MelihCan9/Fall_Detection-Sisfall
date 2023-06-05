import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

# Path to the raw data
data_path = "C:\\Users\\Melih Can\\OneDrive\\Masaüstü\\SisFall\\SisFallDatasetAnnotation-master\\SisFallDatasetAnnotation-master\\SisFall_dataset_csv"
fs = os.listdir(data_path)

# print(len(fs)) # 4500

data = pd.DataFrame()
for f in tqdm(fs):
    file_path = os.path.join(data_path, f)
    df = pd.read_csv(file_path, usecols=[1, 2, 3], names=["Ax", "Ay", "Az"])
    if ('D01' in f) or ('D02' in f) or ('D03' in f) or ('D04' in f):
        df = df[:10000]
    # Reducing the number of data. (Reducing the Hz)
    df = df.loc[::2]

    # Define fixed values for sensor properties
    Sensor1_Resolution = 13
    Sensor1_Range = 16
    g_S1 = (2 * Sensor1_Range / 2 ** Sensor1_Resolution)

    # Convert accelerations to g units by doing the calculations
    df['Ax'] = g_S1 * df['Ax']
    df['Ay'] = g_S1 * df['Ay']
    df['Az'] = g_S1 * df['Az']

    # Calculate SVM (Euler norm) property
    A = []
    for i in range(df.shape[0]):
        A.append(np.sqrt(df.iloc[i]['Ax'] ** 2 + df.iloc[i]['Ay'] ** 2 + df.iloc[i]['Az'] ** 2))
    df['SVM'] = A

    # Calculate HOR (horizontal acceleration)
    A = []
    for i in range(df.shape[0]):
        A.append(np.sqrt(df.iloc[i]['Ay'] ** 2 + df.iloc[i]['Az'] ** 2))
    df['HOR'] = A

    # Calculate VER (Vertical Acceleration)
    A = []
    for i in range(df.shape[0]):
        A.append(np.sqrt(df.iloc[i]['Ax'] ** 2 + df.iloc[i]['Az'] ** 2))
    df['VER'] = A

    # Find the highest and lowest SVM value and index
    max_N = df['SVM'].max()
    max_N_index = df.index[df.SVM == max_N][0]

    min_N = df.SVM.min()
    min_N_index = df.index[df.SVM == min_N][0]

    len_df = len(df)  # df.shape[0]

    # Work with a window of 300 measurements that includes the max index (equivalent to 3 seconds of activity recording)
    if max_N_index - 150 < 0:
        df = df[0: 301]

    else:
        if max_N_index + 150 + 1 > len_df:
            df = df[len_df - 302:len_df - 1]

        else:
            # extract the central window
            df = df[max_N_index - 150: max_N_index + 150 + 1]

    df.index = list(range(df.shape[0]))

    # Calculate statistical features for each sensor axis and derived features
    for c in ['Ax', 'Ay', 'Az', 'SVM', 'VER', 'HOR']:
        df['min_' + c] = df[c].min()
        df['max_' + c] = df[c].max()
        df['mean_' + c] = df[c].mean()
        df['median_' + c] = df[c].median()
        df['var_' + c] = df[c].var()
        df['std_' + c] = df[c].std()
        df['kurtosis_' + c] = df[c].kurtosis()
        df['skewness_' + c] = df[c].skew()
        df['range_' + c] = df[c].max() - df[c].min()

    # Calculate correlations between acceleration sensors
    df['corr_valueXY'] = df["Ax"].corr(df["Ay"])
    df['corr_valueXZ'] = df["Ax"].corr(df["Az"])
    df['corr_valueYZ'] = df["Ay"].corr(df["Az"])

    # Calculate correlations between SVM, VER and HOR features
    df['corr_SVM_VER'] = df["SVM"].corr(df["VER"])
    df['corr_SVM_HOR'] = df["SVM"].corr(df["HOR"])
    df['corr_HOR_VER'] = df["HOR"].corr(df["VER"])

    # Calculate pitch angle
    pitch = []
    for i in range(df.shape[0]):
        pitch.append(math.atan2(-df['Ax'].loc[i], df['Az'].loc[i]))
        # Represented in radians, so that for 12-dimensional features, no feature scaling is required.
    df['pitch'] = pitch

    # Calculate statistical properties of pitch angle
    df['max_pitch'] = df["pitch"].max()
    df['min_pitch'] = df["pitch"].min()
    df['mean_pitch'] = df["pitch"].mean()
    df['median_pitch'] = df["pitch"].median()
    df['var_pitch'] = df["pitch"].var()
    df['std_pitch'] = df["pitch"].std()

    # Create label column (D -> 0, F -> 1)
    df['label'] = f[0:1]
    df['label'] = df['label'].map({'D': 0, 'F': 1})

    # Drop unused columns
    df.drop(['Ax', 'Ay', 'Az', 'SVM', 'HOR', 'VER'], axis=1, inplace=True)
    df = df[0:1]

    data = pd.concat([data, df], axis=0)

# Save generated dataframe to CSV file
data.to_csv('sisfall_preprocessed', index=False)
