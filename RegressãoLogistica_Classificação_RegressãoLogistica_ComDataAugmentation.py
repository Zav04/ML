import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import load
from imblearn.over_sampling import SMOTE
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregamento dos dados (pode ser um novo conjunto de dados para classificação)
data = pd.read_excel('ML.xlsx')
# Removendo a primeira coluna que parece ser um identificador
data = data.drop(data.columns[0], axis=1)

# Dividindo os dados em características (X) e alvo (y)
X = data.drop('Abandono', axis=1)
y = data['Abandono']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Aplicando o SMOTE apenas ao conjunto de treino
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Normalizando os dados - importante aplicar o fit apenas no conjunto de treino
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test) 

# Carregar o modelo treinado
log_reg = load('./Modelos/modelo_LogisticRegression_ComDataAugmentation.joblib')

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



plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Logistic Regression Com Data Augmentation')

