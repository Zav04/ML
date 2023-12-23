import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('input_units', min_value=8, max_value=128, step=16),
                    input_dim=X_train_scaled.shape[1], activation='relu'))
    
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(Dense(hp.Int(f'dense_{i}_units', min_value=8, max_value=128, step=16), activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=500,  # Número de variações de hiperparâmetros a testar
    executions_per_trial=50,  # Número de modelos a treinar por tentativa
    directory='my_dir',  # Diretório para armazenar os logs do keras tuner
    project_name='keras_tuner_demo')

tuner.search(X_train_scaled, y_train, epochs=200, validation_data=(X_test_scaled, y_test))



# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Os melhores hiperparâmetros são:")
print(f" - Número de unidades na primeira camada: {best_hps.get('input_units')}")
print(f" - Número de camadas ocultas: {best_hps.get('n_layers')}")

for i in range(best_hps.get('n_layers')):
    print(f" - Número de unidades na camada {i+1}: {best_hps.get('dense_{i}_units')}")

print(f" - Taxa de aprendizado: {best_hps.get('learning_rate')}")


# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Avaliar o melhor modelo
loss, accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f'Acurácia do melhor modelo: {accuracy:.2f}')