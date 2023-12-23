import pandas as pd

# Carregar os dados
data = pd.read_excel('ML.xlsx')

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
features_to_remove = list(low_correlation_features)  # Adicione a isto conforme necessário

# Remover as variáveis selecionadas do conjunto de dados
data = data.drop(columns=features_to_remove)

print(data)
