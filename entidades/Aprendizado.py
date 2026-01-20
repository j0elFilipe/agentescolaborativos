import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

class ModeloBase:
    """
    CLASSE BASE PARA TODOS OS MODELOS DE MACHINE LEARNING
    """

    def __init__(self, nome):
        self.nome = nome
        self.modelo = None
        self.treinado = False
        self.acuracia = 0

    def treinar(self, X, y):
        """
        TREINA O MODELO COOM DADOS DE ENTRADA 
        """
        raise NotImplementedError("Subclasses devem implementar treinar()")
    
    def prever(self, X):
        """
        FAZ PREDIÇÃO PARA NOVOS DADOS
        """
        if not self.treinado:
            raise Exception(f"Modelo {self.nome} não foi treinado ainda!")
        return self.modelo.predict(X)
    

    def prever_melhor_celula(self, celulas_possiveis, posicao_actual):
        """
        ESCOLHE A MELHOR CÉLULA ENTRE AS POSSÍVEIS
        """
        if not celulas_possiveis:
            return None
        
        #CALCULAR FEATURES PARA CADA CÉLULA POSSÍVEL
        X = []
        for cx, cy in celulas_possiveis:
            dist_centro = np.sqrt((cx - 5)**2 + (cy - 5)**2)
            dist_actual = np.sqrt((cx - posicao_actual[0])**2 + (cy - posicao_actual[1])**2)
            X.append([cx, cy, dist_centro, dist_actual])

        #PREVER PROBABILIDADES
        if hasattr(self.modelo, 'predict_proba'):
            probas = self.modelo.predict_proba(X)

            #ESCOLHER CÉLULA COM MAIOR PROBABILIDADE DE SER TESOURO
            melhor_idx = np.argmax(probas[:, 2]) if probas.shape[1] > 2 else np.argmax(probas[:, 1])
        else:
            predicoes = self.modelo.predict(X)
            #PREFERIR TESOUROS, DEPOIS LIVRES, EVITAR BOMBAS
            prioridades = {'T': 3, 'L': 2, 'B': 1, 'E': 0}
            scores = [prioridades.get(p, 0) for o in predicoes]
            melhor_idx = np.argmax(scores)

        return celulas_possiveis[melhor_idx]
    
    
    
    def salvar_modelo(self, caminho):
        """
        SALVA O MODELO TREINADO NUM ARQUIVO
        """
        with open(caminho, 'wb') as f:
            pickle.dump(self, f)
    

    @staticmethod
    def carregar_modelo(caminho):
        """
        CARREGA UM MODELO SALVO
        """
        with open(caminho, 'rb') as f:
            return pickle.load(f)
        


class ModeloKNN(ModeloBase):
    """
    MODELO K-NEAREST NEIGHBORS PARA CLASSIFICAR AS CÉLULAS
    """
    def __init__(self, n_neighbors = 3):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.modelo = KNeighborsClassifier(n_neighbors = n_neighbors)
        


    def treino_knn(self, X, y):
        """
        TREINA O MODELO KNN
        """

        #DIVIDIR DADOS EM TREINO E TESTE
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #TREINAR
        self.modelo.fit(X_train, y_train)

        #AVALIAR
        y_pred = self.modelo.predict(X_test)
        self.acuracia = accuracy_score(y_test, y_pred)
        self.treinado = True

        return{
             'acuracia': self.acuracia,
            'relatorio': classification_report(y_test, y_pred, zero_division=0)
            }
    
class ModeloArvoreDecisao(ModeloBase):
    """
    MODELO ÁRVORE DE DECISÃO PARA CLASSIFICAÇÃO DE CÉLULAS
    """

    def __init__(self, max_depth = 5):
        super().__init__("Árvore de Decisão")
        self.max_depth = max_depth
        self.modelo = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            min_samples_split=5
        )
    
    def treinar_arvore_decisao(self, X, y):
        """
        TREINA O MODELO DE ÁRVORE DE DECISÃO
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.modelo.fit(X_train, y_train)

        y_pred = self.modelo.predict(X_test)
        self.acuracia = accuracy_score(y_test, y_pred)
        self.treinado = True
        return {
            'acuracia': self.acuracia,
            'relatorio': classification_report(y_test, y_pred, zero_division=0),
            'importancia_features': self.modelo.feature_importances_
        }
    

class ModeloNaiveBayes(ModeloBase):
    """
    MODELO NAIVE BAYES PARA CLASSIFICAÇÃO DE CÉLULAS
    """

    def __init__(self):
        super().__init__("Naive Bayes")
        self.modelo = GaussianNB()
    
    def treinar_naive_bayes(self, X, y):
        """
        TREINA O MODELO NAIVE BAYES
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.modelo.fit(X_train, y_train)

        y_pred = self.modelo.predict(X_test)
        self.acuracia = accuracy_score(y_test, y_pred)
        self.treinado = True

        return{
            'acuracia': self.acuracia,
            'relatorio': classification_report(y_test, y_pred, zero_division=0)
        }
    

    def gerar_dados_treino(num_amostras = 1000, tamanho_ambiente = 10):
        """
        GERA DADOS SINTÉTICOS PARA TREINO DOS MODELOS 
        Returns:
            X: Features (posição x, y, distância do centro, distância da borda)
            y: Labels (L, B, T)
        """
        X = []
        y = []

        for _ in range(num_amostras):
            x = np.random.randint(0, tamanho_ambiente)
            y_coord = np.random.randint(0, tamanho_ambiente)

            #FEATURES
            dist_centro = np.sqrt((x - tamanho_ambiente/2)**2 + (y_coord - tamanho_ambiente/2)**2)
            dist_borda = min(x, y_coord, tamanho_ambiente - 1 - x, tamanho_ambiente - 1 - y_coord)

            #LÓGICA PARA GERAR LABELS (SIMULANDO PADRÕES)
            rand = np.random.random()

            #BOMBAS MAIS PROVÁVEIS NO CENTRO
            if dist_centro < 3:
                if rand < 0.5:
                    label = 'B'
                elif rand < 0.8:
                    label = 'L'
                else:
                    label = 'T'
            #TESOUROS MAIS PROVÁVEIS NAS BORDAS
            elif dist_borda <= 1:
                if rand < 0.4:
                    label = 'T'
            elif rand < 0.7:
                label = 'L'
            else:
                label = 'B'
            
        #ÁREA INTERMÉDIA MAIS CÉLULAS LIVRES
        else:
            if rand < 0.6:
                label = 'L'
            elif rand < 0.8:
                label = 'B'
            else:
                label = 'T'
        
        X.append([x, y_coord, dist_centro, dist_borda])
        y.append([label])
    
        return np.array(X), np.array(y)
    
    
    def treinar_todos_modelos(X = None, y = None):
        """
        TREINAR TODOS OS TRÊS MODELOS E RETORNA RESULTADOS COMPARATIVOS
        """
        if X is None or y is None:
            X, y = gerar_dados_treino(num_amostras = 2000)
        
        modelos = [
            ModeloKNN(n_neighbors=5),
            ModeloArvoreDecisao(max_depth = 8),
            ModeloNaiveBayes()
        ]

        resultados = {}

        for modelo in modelos:
            print(f"\nTreinando {modelo.nome}...")
            resultado = modelo.treinar(X, y)
            resultados[modelo.nome] = {
                'modelo': modelo,
                'acuracia': resultado['acuracia'],
                'detalhes': resultado
            }
            print(f"Acuracia: {resultado['acuracia']:.4f}")
        
        return resultados
    
    def comparar_modelos(resultados):
        """
        COMPRA O DESEMPENHO  DOS MODELOS TREINADOS
        """
        print("\n +" + "="*60)
        print("COMPARAÇÃO DE MODELOS")
        print("="*60)

        for nome, info in resultados.items():
            print(f"\n{nome}:")
            print(f" Acuracia: {info['acuracia']:.4f}")
            if 'importancia_features' in info['detalhes']:
                print(f"  Importância Features: {info['detalhes']['importancia_features']}")
        
        #DETERMINA MELHOR MODELO
        melhor = max(resultados.items(), key = lambda x: x[1]['acuracia'])
        print(f"\n Melhor Modelo: {melhor[0]} (Acurácia: {melhor[1]['acuracia']:.4f})")
        
        return melhor[1]['modelo']
    
