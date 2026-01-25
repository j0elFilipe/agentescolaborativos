import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

class ModeloBase:
    """
    Classe base para todos os modelos de Machine Learning.
    """
    
    def __init__(self, nome):
        self.nome = nome
        self.modelo = None
        self.treinado = False
        self.acuracia = 0
    
    def treinar(self, X, y):
        """
        Treina o modelo com dados de entrada.
        
        Args:
            X: Features (array 2D)
            y: Labels (array 1D)
        
        Returns:
            dict: Métricas de treinamento
        """
        raise NotImplementedError("Subclasses devem implementar treinar()")
    
    def prever(self, X):
        """
        Faz predição para novos dados.
        
        Args:
            X: Features para predição
            
        Returns:
            array: Predições
        """
        if not self.treinado:
            raise Exception(f"Modelo {self.nome} não foi treinado ainda!")
        return self.modelo.predict(X)
    
    def escolher_melhor_celula(self, celulas_possiveis):
        """
        Escolhe a melhor célula entre as possíveis usando o modelo ML.
        
        Este é o método CHAVE que faz os agentes usarem ML!
        Compatível com main.py (usa 3 features)
        
        Args:
            celulas_possiveis: Lista de tuplas (x, y)
            
        Returns:
            tuple: (x, y) da melhor célula escolhida
        """
        if not celulas_possiveis:
            return None
        
        melhor_celula = None
        melhor_score = -1000
        
        for cx, cy in celulas_possiveis:
            # Calcular features (3 features como no main.py)
            dist_centro = np.sqrt((cx - 5)**2 + (cy - 5)**2)
            
            # Prever tipo de célula usando o modelo ML
            predicao = self.modelo.predict([[cx, cy, dist_centro]])[0]
            
            # Sistema de pontuação baseado na predição
            score = 0
            if predicao == 'T':  # Tesouro é o melhor
                score = 100
            elif predicao == 'L':  # Livre é neutro
                score = 50
            elif predicao == 'B':  # Bomba é ruim
                score = -50
            
            # Adicionar aleatoriedade para exploração (10% de variação)
            score += np.random.randint(-10, 10)
            
            if score > melhor_score:
                melhor_score = score
                melhor_celula = (cx, cy)
        
        return melhor_celula if melhor_celula else celulas_possiveis[0]
    
    def salvar_modelo(self, caminho):
        """
        Salva o modelo treinado em arquivo.
        
        Args:
            caminho: Path do arquivo
        """
        with open(caminho, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def carregar_modelo(caminho):
        """
        Carrega um modelo salvo.
        
        Args:
            caminho: Path do arquivo
            
        Returns:
            ModeloBase: Modelo carregado
        """
        with open(caminho, 'rb') as f:
            return pickle.load(f)


class ModeloKNN(ModeloBase):
    """
    Modelo K-Nearest Neighbors para classificação de células.
    
    Funciona bem para padrões espaciais locais.
    """
    
    def __init__(self, n_neighbors=5):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def treinar(self, X, y):
        """
        Treina o modelo KNN.
        
        Args:
            X: Features (posição x, y, distância centro, distância borda)
            y: Labels ('L', 'B', 'T')
        """
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar
        self.modelo.fit(X_train, y_train)
        
        # Avaliar
        y_pred = self.modelo.predict(X_test)
        self.acuracia = accuracy_score(y_test, y_pred)
        self.treinado = True
        
        return {
            'acuracia': self.acuracia,
            'relatorio': classification_report(y_test, y_pred, zero_division=0)
        }


class ModeloArvoreDecisao(ModeloBase):
    """
    Modelo Árvore de Decisão para classificação de células.
    
    Funciona bem para regras claras (ex: se dist_centro < X então bomba).
    """
    
    def __init__(self, max_depth=8):
        super().__init__("Árvore de Decisão")
        self.max_depth = max_depth
        self.modelo = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            min_samples_split=5
        )
    
    def treinar(self, X, y):
        """
        Treina o modelo de Árvore de Decisão.
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
    Modelo Naive Bayes para classificação de células.
    
    Funciona bem com dados probabilísticos.
    """
    
    def __init__(self):
        super().__init__("Naive Bayes")
        self.modelo = GaussianNB()
    
    def treinar(self, X, y):
        """
        Treina o modelo Naive Bayes.
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
            'relatorio': classification_report(y_test, y_pred, zero_division=0)
        }


def gerar_dados_treino(num_amostras=2000, tamanho_ambiente=10):
    """
    Gera dados sintéticos para treino dos modelos.
    
    Simula padrões realistas:
    - Bombas mais frequentes no centro
    - Tesouros mais frequentes nas bordas
    - Células livres distribuídas
    
    COMPATÍVEL COM MAIN.PY: Usa 3 features (x, y, dist_centro)
    
    Args:
        num_amostras: Quantidade de exemplos
        tamanho_ambiente: Tamanho do grid (padrão 10x10)
    
    Returns:
        tuple: (X, y) onde X são features e y são labels
    """
    X = []
    y = []
    
    for _ in range(num_amostras):
        x = np.random.randint(0, tamanho_ambiente)
        y_coord = np.random.randint(0, tamanho_ambiente)
        
        # Feature principal
        dist_centro = np.sqrt((x - tamanho_ambiente/2)**2 + (y_coord - tamanho_ambiente/2)**2)
        
        # Lógica para gerar labels (simulando padrões)
        rand = np.random.random()
        
        # Bombas mais prováveis no centro
        if dist_centro < 3:
            if rand < 0.5:
                label = 'B'
            elif rand < 0.8:
                label = 'L'
            else:
                label = 'T'
        # Tesouros mais prováveis longe do centro
        elif dist_centro > 6:
            if rand < 0.4:
                label = 'T'
            elif rand < 0.7:
                label = 'L'
            else:
                label = 'B'
        # Área intermediária: mais células livres
        else:
            if rand < 0.6:
                label = 'L'
            elif rand < 0.8:
                label = 'B'
            else:
                label = 'T'
        
        # USAR 3 FEATURES (compatível com main.py)
        X.append([x, y_coord, dist_centro])
        y.append(label)
    
    return np.array(X), np.array(y)


def treinar_todos_modelos(X=None, y=None, verbose=True):
    """
    Treina todos os três modelos e retorna resultados comparativos.
    
    Args:
        X: Features (opcional, gera automaticamente se None)
        y: Labels (opcional, gera automaticamente se None)
        verbose: Se True, imprime informações de treino
    
    Returns:
        dict: Dicionário com os 3 modelos treinados
    """
    if X is None or y is None:
        if verbose:
            print("Gerando dados de treino...")
        X, y = gerar_dados_treino(num_amostras=2000)
        if verbose:
            print(f"✓ Gerados {len(X)} exemplos")
    
    modelos = {
        'knn': ModeloKNN(n_neighbors=5),
        'tree': ModeloArvoreDecisao(max_depth=8),
        'bayes': ModeloNaiveBayes()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        if verbose:
            print(f"\nTreinando {modelo.nome}...")
        resultado = modelo.treinar(X, y)
        resultados[nome] = {
            'modelo': modelo,
            'acuracia': resultado['acuracia'],
            'detalhes': resultado
        }
        if verbose:
            print(f"✓ Acurácia: {resultado['acuracia']:.4f}")
    
    return resultados


def comparar_modelos(resultados):
    """
    Compara o desempenho dos modelos treinados.
    
    Args:
        resultados: Dict retornado por treinar_todos_modelos()
    
    Returns:
        ModeloBase: Melhor modelo baseado em acurácia
    """
    print("\n" + "="*60)
    print("COMPARAÇÃO DE MODELOS")
    print("="*60)
    
    for nome, info in resultados.items():
        print(f"\n{nome.upper()}:")
        print(f"  Acurácia: {info['acuracia']:.4f}")
        if 'importancia_features' in info['detalhes']:
            feat_names = ['x', 'y', 'dist_centro', 'dist_borda']
            print(f"  Importância Features:")
            for fname, imp in zip(feat_names, info['detalhes']['importancia_features']):
                print(f"    {fname}: {imp:.4f}")
    
    # Determinar melhor modelo
    melhor = max(resultados.items(), key=lambda x: x[1]['acuracia'])
    print(f"\n🏆 Melhor Modelo: {melhor[0].upper()} (Acurácia: {melhor[1]['acuracia']:.4f})")
    
    return melhor[1]['modelo']


