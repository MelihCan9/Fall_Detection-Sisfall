import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


from keras import models, layers, callbacks, regularizers

model = models.Sequential()
model.add(layers.Dense(units=64, input_shape=(X_train.shape[1],), activation='relu',
                       kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))  # Dropout layer with a dropout rate of 0.5
model.add(layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))  # Dropout layer with a dropout rate of 0.5
model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))  # Dropout layer with a dropout rate of 0.5
model.add(layers.Dense(units=1, activation='sigmoid'))

# model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

callback_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='auto')]

history = model.fit(
    partial_x_train, partial_y_train,
    epochs=300, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callback_list
)

score = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

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

# Plotting the loss and accuracy plots of train and validation sets.
"""history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Train Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Train Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Train and Validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()"""

from sklearn.metrics import classification_report, confusion_matrix, f1_score

threshold = 0.65


predictions = model.predict(X_test_scaled)
binary_predictions = (predictions > threshold).astype(int)

cm = confusion_matrix(y_test, binary_predictions)
# print(cm)

TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

SE = TP / (TP + FN)
SP = TN / (TN + FP)
AC = (SE + SP) / 2

print("SE:{} SP:{} AC:{} ".format(SE, SP, AC))

# SE:0.9851851851851852 SP:0.9694444444444444 AC:0.9773148148148147 | Test accuracy: 0.9755555391311646






