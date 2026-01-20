import numpy as np
import random

class Ambiente:
    """
    CLASSE QUE REPRESENTA O  AMBIENTE DE EXPLORAÇÃO 10x10
    """

    def __init__(self, tamanho = 10, perc_livres = 50, perc_bombas = 30, perc_tesouros = 20):
        """
        INICIALIZA O AMBIENTE
        """
        self.tamanho = tamanho
        self.perc_livres = perc_livres
        self.perc_bombas = perc_bombas
        self.perc_tesouros = perc_tesouros
        self.matriz = None
        self.tesouros_iniciais = 0
        self.bombas_iniciais = 0
        self.criar_ambiente()

    #FUNÇÃO PARA CRIAR O AMBIENTE
    def criar_ambiente(self):
        #CRIAR MATRIZ COM DISTRIBUIÇÃO DE PROBABILIDADES
        probabilidades = [
            self.perc_livres / 100,
            self.perc_bombas / 100,
            self.perc_tesouros / 100
        ]
        self.matriz = np.random.choice(
            ['L', 'B', 'T'],
            size = (self.tamanho, self.tamanho),
            p = probabilidades
        )
        
        #ADICIONAR BANDEIRA EM POSIÇÃO ALEATÓRIA
        fx = random.randint(0, self.tamanho - 1)
        fy = random.randint(0, self.tamanho - 1)
        self.matriz[fx, fy] = 'F'
        self.bandeira_pos = (fx, fy)

        #CONTAR RECURSOS INICIAIS
        self.tesouros_iniciais = np.sum(self.matriz == 'T')
        self.bombas_iniciais = np.sum(self.matriz == 'B')

        return self.matriz
    

    def ajustar_proporcao_bombas(self, nova_perc_bombas):
        """
        AJUSTAR A PROPORÇÃO DE BOMBAS NO AMBIENTE
        """
        self.perc_bombas = nova_perc_bombas
        self.perc_livres = 100 - nova_perc_bombas - self.perc_tesouros
        self.criar_ambiente()
    

    def get_celula(self, x, y):
        """
        RETORNA O CONTEÚDO DE UMA CÉLULA
        """
        if 0 <= x < self.tamanho and 0 <= y < self.tamanho:
            return self.matriz[x, y]
        return None
    
    
    def set_celula(self, x, y, valor):
        """
        DEFINE O VALOR DE UMA CÉLULA
        """
        if 0 <= x < self.tamanho and 0 <= y < self.tamanho:
            self.matriz[x, y] = valor
            return True
        return False
    

    def get_vizinhos(self, x, y, incluir_diagonais = True):
        """
        RETORNA AS CÉLULAS VIZINHAS DE UMA POSIÇÃO
        """
        vizinhos = []
        movimentos = [(0, 1), (0, 1), (0, -1), (-1, 0)]
        if incluir_diagonais:
            movimentos.extend = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in movimentos:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.tamanho and 0 <= ny < self.tamanho:
                vizinhos.append({
                    'pos': (nx, ny),
                    'conteudo': self.matriz[nx, ny]
                })
        
        return vizinhos
    

    def contar_bombas_adjacentes(self, x, y):
        """
        CONTA QUANTAS BOMBAS EXISTEM NAS CÉLULAS ADJACENTES
        """
        vizinhos = self.get_vizinhos(x, y)
        return sum(1 for v in vizinhos if v['conteudo'] == 'B')
    

    def contar_tesouros_adjacentes(self, x, y):
        """
        CONTA QAUANTOS TESOUROS EXISTEM NAS CÉLULAS ADJACENTES
        """
        vizinhos = self.get_vizinhos(x, y)
        return sum(1 for v in vizinhos if v['conteudo'] == 'T')
    
    
    def get_estatisticas(self):
        """
        RETORNA ESTATÍSTICAS DO AMBIENTE
        """
        return {
            'tamanho': self.tamanho,
            'livres': np.sum(self.matriz == 'L'),
            'bombas': np.sum(self.matriz == 'B'),
            'tesouros': np.sum(self.matriz == 'T'),
            'exploradas': np.sum(self.matriz == 'E'),
            'bandeira': self.bandeira_pos,
            'total_celulas': self.tamanho * self.tamanho,
            'tesouros_iniciais': self.tesouros_iniciais,
            'bombas_iniciais': self.bombas_iniciais
        }
    

    def ambiente_completamente_explorado(self):
        """
        VEIRIFICA SE TODAS AS CÉLULAS FORAM EXPLORADAS
        """
        elementos_nao_explorados = ['L', 'B', 'T']
        return not np.any(np.isin(self.matriz, elementos_nao_explorados))
    

    def reset(self):
        """
        REINICIA O AMBIENTE PARA UM NOVO ESTADO
        """
        self.criar_ambiente()

    
    def exportar_para_treino(self):
        """
        EXPORTA DADOS DO AMBIENTE EM FORMATO ADEQUADO PARA TREINO DE ML.
        RETORNA UMA LISTA DE FEATURES PARA CADA CÉLULA
        """
        dados = []
        for i in range(self.tamanho):
            for j in range(self.tamanho):
                feature = {
                    'x': i,
                    'y': j,
                    'dist_centro': np.sqrt((i - self.tamanho/2)**2 + (j - self.tamanho/2)**2),
                    'bombas_adj': self.contar_bombas_adjacentes(i, j),
                    'tesouros_adj': self.contar_tesouros_adjacentes(i, j),
                    'tipo': self.matriz[i, j]
                }
                dados.append(feature)
        return dados
    
    def visualizar_terminal(self):
        """
        EXIBE O AMBIENTE NO TERMINAL (ÚTIL PARA DEUG)
        """
        simbolos = {
            'L': '⬜',
            'B': '💣',
            'T': '💎',
            'F': '🏁',
            'E': '⬛'
        }

        print("\n ", end="")
        for j in range(self.tamanho):
            print(f"{j:2}",end=" ")
        print()

        for i in range(self.tamanho):
            print(f"{i:2} ", end="")
            for j in range(self.tamanho):
                simbolo = simbolos.get(self.matriz[i, j], '?')
                print(simbolo, end=" ")
            print()
        print()

    
    def clonar(self):
        """
        CRIA UMA CÓPIA DO AMBIENTE
        """
        novo_ambiente = Ambiente(
            tamanho = self.tamanho,
            perc_livres = self.perc_livres,
            perc_bombas = self.perc_bombas,
            perc_tesouros = self.perc_tesouros
        )

        novo_ambiente.matriz = self.matriz.copy()
        novo_ambiente.tesouros_iniciais = self.tesouros_iniciais
        novo_ambiente.bombas_iniciais = self.bombas_iniciais
        novo_ambiente.bandeira_pos = self.bandeira_pos
        return novo_ambiente
    

    def __repr__(self):
        stats = self.get_estatisticas()
        return (f"Ambiente ({self.tamanho}x{self.tamanho}, " 
                f"L:{stats['livres']}, B: {stats['bombas']}, "
                f"T:{stats['tesouros']}, E:{stats['exploradas']})")
    

    def criar_conjunto_ambientes(num_ambientes = 10, variacao_bombas = True):
        """
        CRIA MÚLTIPLOS AMBIENTES PARA TESTES
        ARGS:
            num_ambientes: NÚMERO DE AMBIENTES
            variacao_bombas: SE TRUE, VARIA A % DE BOMBAS ENTRE 20% E 80%

            Returns:
                LISTA DE AMBIENTES
        """
        ambientes = []

        if variacao_bombas:
            percentuais = np.linspace(20, 80, num_ambientes)
        else:
            percentuais = [50] * num_ambientes

        for perc_bomb in percentuais:
            perc_livre = 100 - perc_bomb - 10 #10% TESOUROS FIXO
            ambiente = Ambiente(
                tamanho = 10,
                perc_livres = perc_livre,
                perc_bombas = perc_bomb,
                perc_tesouros = 10
            )
            ambientes.append(ambiente)

        return ambientes