import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv('dataset/sisfall_preprocessed')

# print(pd.value_counts(df['label']))

# Reduced the number of columns from 68 to 16(+1)
cols = ['max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax',
        'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
        'max_Az', 'min_Az', 'var_Az', 'mean_Az',
        # 'max_VER', 'min_VER', 'var_VER', 'mean_VER',
        'max_pitch', 'min_pitch', 'var_pitch', 'mean_pitch', 'label']

df = df[cols]
# print(df.info())

y = df['label']
X = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initializing Validation Datasets (it should be modified if used with mini (sisfall_500) data)
X_val = X_train_scaled[:750]
partial_x_train = X_train_scaled[750:]

y_val = y_train[:750]
partial_y_train = y_train[750:]

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC

model = SVC(C=100, kernel='rbf')
# model.fit(partial_x_train, partial_y_train)
model.fit(X_train_scaled, y_train)  # For one last training w/o validation, after tuning the hyperparameters.

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", accuracy)

# Optional
"""
# Create a sample test value
test_value = X_test_scaled.iloc[42]

# Reshaping the test value according to the expectation of the model
test_value = test_value.values.reshape(1, -1)

# Making predictions using the model, and used argmax to see which class is sample belongs.
predictions = np.argmax(model.predict(test_value), axis=-1)

print(predictions)
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
# SE:0.987012987012987 SP:0.9695290858725761 AC:0.9782710364427816 | Test accuracy: 0.98

