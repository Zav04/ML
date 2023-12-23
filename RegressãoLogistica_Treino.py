import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregamento dos dados
data = pd.read_excel('ML.xlsx')


# Verificar o balanceamento das classes
class_balance = data['Abandono'].value_counts(normalize=True)
print("Balanceamento das classes:\n", class_balance)

# Resumo estatístico das características numéricas
stats_summary = data.describe()
print("Resumo estatístico das características:\n", stats_summary)


# Calcular a matriz de correlação
correlation_matrix = data.corr()

# Exibir a correlação com a variável alvo
correlation_with_target = correlation_matrix['Abandono'].sort_values(ascending=False)
print(correlation_with_target)


# Remover variáveis com correlação muito baixa
low_correlation_threshold = 0.01  # Ajuste este valor conforme necessário
high_correlation_threshold = 0.9  # Ajuste este valor conforme necessário

# Encontrar características com baixa correlação com a variável alvo
low_correlation_features = correlation_with_target[abs(correlation_with_target) < low_correlation_threshold].index

# Encontrar pares de características altamente correlacionadas
high_correlation_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and abs(correlation_matrix.loc[i, j]) > high_correlation_threshold]

# Decidir quais variáveis remover baseado em conhecimento do domínio e correlações
features_to_remove = list(low_correlation_features)

# Remover as variáveis selecionadas do conjunto de dados
data = data.drop(columns=features_to_remove)

# Dividindo os dados em características (X) e alvo (y)
X = data.drop('Abandono', axis=1)
y = data['Abandono']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de regressão logística com hiperparâmetros customizados
log_reg = LogisticRegression(C=0.1, solver='liblinear', max_iter=100, penalty='l1')
log_reg.fit(X_train_scaled, y_train)  # Use o conjunto de treino aumentado

# Guardar o modelo treinado
dump(log_reg, './Modelos/modelo_LogisticRegression.joblib')

# Avaliação do modelo
predictions = log_reg.predict(X_test_scaled)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')



plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Logistic Regression')
