import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


data = pd.read_excel('ML.xlsx')



class_balance = data['Abandono'].value_counts(normalize=True)
print("Balanceamento das classes:\n", class_balance)


stats_summary = data.describe()
print("Resumo estatístico das características:\n", stats_summary)


X = data.drop('Abandono', axis=1)
y = data['Abandono']


correlation_matrix = data.corr()


correlation_with_target = correlation_matrix['Abandono'].sort_values(ascending=False)
print(correlation_with_target)



low_correlation_threshold = 0.1  
high_correlation_threshold = 0.9  


low_correlation_features = correlation_with_target[abs(correlation_with_target) < low_correlation_threshold].index


high_correlation_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and abs(correlation_matrix.loc[i, j]) > high_correlation_threshold]


features_to_remove = list(low_correlation_features)


data = data.drop(columns=features_to_remove)



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




log_reg = LogisticRegression(C=0.1, solver='liblinear', max_iter=100, penalty='l1')
log_reg.fit(X_train_scaled, y_train)  


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



train_sizes, train_scores, test_scores = learning_curve(estimator=log_reg,
                                                        X=X_balanced, y=y_balanced,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=5,
                                                        n_jobs=-1,
                                                        scoring='accuracy')


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)


test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Treino')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validação')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.title('Curvas de Aprendizagem - Regressão Logista')
plt.xlabel('Tamanho do Conjunto de Treino')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()
