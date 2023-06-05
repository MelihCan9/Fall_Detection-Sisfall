import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, mean_absolute_error

df = pd.read_csv('dataset/sisfall_preprocessed')

# print(pd.value_counts(df['label']))

# Reduced the number of columns from 68 to 16(+1)
cols = ['max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax',
        'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
        'max_Az', 'min_Az', 'var_Az', 'mean_Az',
        'max_VER', 'min_VER', 'var_VER', 'mean_VER',
        'max_pitch', 'min_pitch', 'var_pitch', 'mean_pitch', 'label']

df = df[cols]
# print(df.info())

y = df['label']
X = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
# STD
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# K-Fold Cross Validation Version (Optional)
"""
def build_model():
    model = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    return model

k = 4
num_val_samples = len(X_train_scaled) // k
num_epochs = 80
all_se_history = []
all_sp_history = []
all_ac_history = []

test_mse_score, test_mae_score = 0,0
for i in range(k):
    print("processing fold #", i)

    val_data = X_train_scaled[i * num_val_samples: (i+1) * num_val_samples]
    val_label = y_train[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate([X_train_scaled[i * num_val_samples:],
                                         X_train_scaled[:(i+1) * num_val_samples]],
                                        axis=0)

    partial_train_label = np.concatenate([y_train[i * num_val_samples:],
                                          y_train[: (i+1) * num_val_samples]],
                                         axis=0)

    model = build_model()

    model.fit(partial_train_data, partial_train_label)

    predictions = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, predictions)
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    AC = (SE + SP) / 2

    all_se_history.append(SE)
    all_sp_history.append(SP)
    all_ac_history.append(AC)

average_se = np.mean(all_se_history)
average_sp = np.mean(all_sp_history)
average_ac = np.mean(all_ac_history)

print("Average Sensitivity (SE):", average_se)
print("Average Specificity (SP):", average_sp)
print("Average Accuracy (AC):", average_ac)
"""

# Initializing Validation Datasets (it should be modified if used with mini (sisfall_500) data)
X_val = X_train_scaled[:750]
partial_x_train = X_train_scaled[750:]

y_val = y_train[:750]
partial_y_train = y_train[750:]

model = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)

# model.fit(partial_x_train, partial_y_train)
model.fit(X_train_scaled, y_train)  # For one last training w/o validation, after tuning the hyperparameters.

predictions = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", accuracy)

# Optional
"""
# Create a sample test value
#test_value = X_test_scaled.iloc[42]

# Reshaping the test value according to the expectation of the model
#test_value = test_value.values.reshape(1, -1)

# Making predictions using the model, and used argmax to see which class is sample belongs.
#predictions = np.argmax(model.predict(test_value), axis=-1)

#print(predictions)
"""

# print(confusion_matrix(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

SE = TP / (TP + FN)
SP = TN / (TN + FP)
AC = (SE + SP) / 2

print("SE:{} SP:{} AC:{} ".format(SE, SP, AC))

# w/o val:
# SE:0.9853479853479854 SP:0.9858757062146892 AC:0.9856118457813373 | Test accuracy: 0.9855555555555555

