import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregamento dos dados
data = pd.read_excel('ML.xlsx')

# Pré-processamento: exemplo simples
# Removendo a primeira coluna que parece ser um identificador
data = data.drop(data.columns[0], axis=1)

# Dividindo os dados em características (X) e alvo (y)
X = data.drop('Abandono', axis=1)
y = data['Abandono']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Normalizando os dados
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

# Construindo o modelo de rede neural
model = Sequential()
model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)

# Avaliando o modelo
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
