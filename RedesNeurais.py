import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_excel('ML.xlsx')


X = data.drop('Abandono', axis=1)
y = data['Abandono']


print("Balanceamento das classes original:\n", y.value_counts(normalize=True))


class_0 = data[data['Abandono'] == 0]
class_1 = data[data['Abandono'] == 1]


class_0_sample = class_0.sample(500, random_state=42) 
class_1_sample = class_1.sample(500, random_state=42) 


balanced_data = pd.concat([class_0_sample, class_1_sample])


balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


X_balanced = balanced_data.drop('Abandono', axis=1)
y_balanced = balanced_data['Abandono']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs disponíveis:", gpus)
else:
    print("GPU não disponível.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model = Sequential()
model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)


_, accuracy = model.evaluate(X_test_scaled, y_test)


nn_predictions = model.predict(X_test_scaled)
nn_predictions = [1 if x > 0.5 else 0 for x in nn_predictions]

accuracy = accuracy_score(y_test, nn_predictions)
precision = precision_score(y_test, nn_predictions)
recall = recall_score(y_test, nn_predictions)
f1 = f1_score(y_test, nn_predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')


plot_confusion_matrix(y_test, nn_predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Neural Network')


