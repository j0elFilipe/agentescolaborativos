from sklearn.neighbors import KNeighborsClassifier

def treino_knn(dados_treino, classes):
    #TREINA O MODELO KNN COM DADOS DE ENTRADA

    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(dados_treino, classes)
    return modelo_knn

def prever_celula(modelo, x, y):
    #FAZ A PREDIÇÃO DE UMA CÉLULA COM BASE NAS SUAS COORDENADAS

    return modelo.predict([[x, y]])[0]

