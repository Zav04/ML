from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Carregamento dos dados
data = pd.read_excel('ML.xlsx')

# Pré-processamento: exemplo simples
# Removendo a primeira coluna que parece ser um identificador
data = data.drop(data.columns[0], axis=1)

# Dividindo os dados em características (X) e alvo (y)
X = data.drop('Abandono', axis=1)
y = data['Abandono']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define um pipeline com SMOTE e XGBoost
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define os parâmetros para o GridSearchCV
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 4, 5],
    'xgb__subsample': [0.5, 0.7, 1.0],
    'xgb__colsample_bytree': [0.5, 0.7, 1.0],
}

# Cria o objeto GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# Realiza a pesquisa com validação cruzada
grid_search.fit(X_train_scaled, y_train)

# Melhores parâmetros encontrados
best_params = grid_search.best_params_

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Imprime os melhores parâmetros
print("Melhores parâmetros:", best_params)
