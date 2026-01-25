import numpy as np

class Agente:
    """
    Classe que representa um agente inteligente no ambiente de exploração.
    Cada agente possui um modelo de ML para tomar decisões.
    """
    
    def __init__(self, id, x, y, ambiente, modelo_tipo='random', modelo_ml=None):
        """
        Inicializa um agente.
        
        Args:
            id: Identificador único do agente (0, 1, 2...)
            x: Posição inicial linha
            y: Posição inicial coluna
            ambiente: Referência à matriz do ambiente
            modelo_tipo: Tipo do modelo ('knn', 'tree', 'bayes', 'random')
            modelo_ml: Instância do modelo ML treinado (opcional)
        """
        self.id = id
        self.x = x
        self.y = y
        self.posicao = (x, y)  # Mantém compatibilidade
        self.ambiente = ambiente
        self.tesouros = 0
        self.bombas_desativadas = 0
        self.vivo = True
        self.conhecimento_compartilhado = set()  # Células conhecidas por todos
        self.historico_movimentos = []
        self.modelo_tipo = modelo_tipo  # 'knn', 'tree', 'bayes'
        self.modelo_ml = modelo_ml  # Modelo ML real para decisões
        self.movimentos = 0  # Contador total de movimentos
        
    def explorar(self, x, y):
        """
        Explora uma célula específica do ambiente.
        
        Returns:
            dict: Resultado da exploração com status, mensagem, etc.
        """
        if not self.vivo:
            return {"status": "morto", "mensagem": f"Agente {self.id} está morto"}
        
        # Verificar limites
        if not (0 <= x < self.ambiente.shape[0] and 0 <= y < self.ambiente.shape[1]):
            return {"status": "invalido", "mensagem": "Movimento fora dos limites"}
        
        conteudo = self.ambiente[x, y]
        self.posicao = (x, y)
        self.historico_movimentos.append((x, y))
        self.movimentos += 1
        
        resultado = {"status": "explorado", "celula": conteudo, "posicao": (x, y)}
        
        # Atualizar posição após exploração
        self.x = x
        self.y = y
        self.posicao = (x, y)
        
        # Processar célula baseado no conteúdo
        if conteudo == 'B':
            if self.bombas_desativadas > 0:
                self.bombas_desativadas -= 1
                self.ambiente[x, y] = 'E'
                resultado["mensagem"] = f"Agente {self.id} desativou bomba em ({x},{y})"
                resultado["acao"] = "desativou_bomba"
            else:
                self.vivo = False
                self.ambiente[x, y] = 'E'
                resultado["mensagem"] = f"Agente {self.id} foi destruído em ({x},{y})"
                resultado["acao"] = "destruido"
                resultado["status"] = "morto"
        
        elif conteudo == 'T':
            self.tesouros += 1
            self.bombas_desativadas += 1
            self.ambiente[x, y] = 'E'
            resultado["mensagem"] = f"Agente {self.id} encontrou tesouro em ({x},{y})"
            resultado["acao"] = "tesouro"
        
        elif conteudo == 'F':
            resultado["mensagem"] = f"Agente {self.id} encontrou a bandeira em ({x},{y})"
            resultado["acao"] = "bandeira"
        
        elif conteudo == 'L':
            self.ambiente[x, y] = 'E'
            resultado["mensagem"] = f"Agente {self.id} explorou célula livre em ({x},{y})"
            resultado["acao"] = "explorou"
        
        # Adicionar ao conhecimento compartilhado
        self.conhecimento_compartilhado.add((x, y))
        
        return resultado
    
    def escolher_proxima_celula(self):
        """
        Escolhe a próxima célula a explorar.
        
        USA O MODELO ML SE DISPONÍVEL, caso contrário escolhe aleatoriamente.
        
        Returns:
            tuple: (x, y) da próxima célula ou None se não houver opções
        """
        x, y = self.x, self.y  # Usar x,y diretos em vez de posicao
        
        # Todos os movimentos possíveis (8 direções)
        movimentos_possiveis = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # Adjacentes
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # Diagonais
        ]
        
        # Filtrar movimentos válidos
        movimentos_validos = []
        for mx, my in movimentos_possiveis:
            if (0 <= mx < self.ambiente.shape[0] and 
                0 <= my < self.ambiente.shape[1] and
                (mx, my) not in self.conhecimento_compartilhado):
                movimentos_validos.append((mx, my))
        
        if not movimentos_validos:
            # Tentar qualquer célula não explorada no grid inteiro
            for i in range(self.ambiente.shape[0]):
                for j in range(self.ambiente.shape[1]):
                    if (i, j) not in self.conhecimento_compartilhado:
                        return (i, j)
            return None
        
        # ========== USAR MODELO ML PARA DECISÃO ========== #
        if self.modelo_ml:
            # Modelo ML escolhe a melhor célula
            melhor_celula = self.modelo_ml.escolher_melhor_celula(movimentos_validos)
            return melhor_celula
        # ================================================= #
        
        # Fallback: escolher aleatoriamente
        return movimentos_validos[np.random.randint(len(movimentos_validos))]
    
    def compartilhar_conhecimento(self, outros_agentes):
        """
        Compartilha o conhecimento com outros agentes.
        
        Args:
            outros_agentes: Lista de outros agentes
        """
        for agente in outros_agentes:
            if agente.id != self.id:
                agente.conhecimento_compartilhado.update(self.conhecimento_compartilhado)
    
    def get_estado(self):
        """
        Retorna o estado atual do agente.
        
        Returns:
            dict: Dicionário com todas as informações do agente
        """
        return {
            "id": self.id,
            "posicao": self.posicao,
            "tesouros": self.tesouros,
            "bombas_desativadas": self.bombas_desativadas,
            "vivo": self.vivo,
            "celulas_exploradas": len(self.conhecimento_compartilhado),
            "historico": len(self.historico_movimentos),
            "modelo_tipo": self.modelo_tipo,
            "movimentos": self.movimentos
        }
    
    def __repr__(self):
        status = "Vivo" if self.vivo else "Morto"
        return f"Agente({self.id}, {self.modelo_tipo.upper()}, {status}, Pos: {self.posicao}, Tesouros: {self.tesouros})"


class GrupoAgentes:
    """
    Gerencia um grupo de agentes colaborativos.
    Facilita sincronização de conhecimento e estatísticas.
    """
    
    def __init__(self):
        self.agentes = {}
        self.conhecimento_global = set()
        # PADRONIZADO: mesmo nome que main.py
        self.estatisticas_modelos = {
            'knn': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []},
            'tree': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []},
            'bayes': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []}
        }
    
    def registrar_agente(self, agente):
        """
        Registra um novo agente no grupo.
        
        Args:
            agente: Instância de Agente
            
        Returns:
            bool: True se registrado com sucesso
        """
        if agente.id not in self.agentes:
            self.agentes[agente.id] = agente
            # Registrar nas estatísticas por modelo
            if agente.modelo_tipo in self.estatisticas_modelos:
                self.estatisticas_modelos[agente.modelo_tipo]['agentes'].append(agente.id)
            return True
        return False
    
    def sincronizar_conhecimento(self):
        """
        Sincroniza o conhecimento entre todos os agentes.
        Todos os agentes terão acesso ao conhecimento global.
        """
        # Coletar todo conhecimento
        for agente in self.agentes.values():
            self.conhecimento_global.update(agente.conhecimento_compartilhado)
        
        # Distribuir para todos
        for agente in self.agentes.values():
            agente.conhecimento_compartilhado = self.conhecimento_global.copy()
    
    def get_agentes_vivos(self):
        """
        Retorna lista de agentes vivos.
        
        Returns:
            list: Lista de agentes vivos
        """
        return [ag for ag in self.agentes.values() if ag.vivo]
    
    def atualizar_estatisticas(self):
        """
        Atualiza estatísticas de performance por modelo ML.
        """
        # Resetar contadores
        for modelo in self.estatisticas_modelos:
            self.estatisticas_modelos[modelo]['tesouros'] = 0
            self.estatisticas_modelos[modelo]['mortes'] = 0
            self.estatisticas_modelos[modelo]['movimentos'] = 0
        
        # Contar estatísticas
        for agente in self.agentes.values():
            if agente.modelo_tipo in self.estatisticas_modelos:
                stats = self.estatisticas_modelos[agente.modelo_tipo]
                stats['tesouros'] += agente.tesouros
                stats['movimentos'] += agente.movimentos
                if not agente.vivo:
                    stats['mortes'] += 1
    
    def get_estatisticas(self):
        """
        Retorna estatísticas gerais e por modelo.
        
        Returns:
            dict: Estatísticas completas
        """
        self.atualizar_estatisticas()
        
        agentes_vivos = len(self.get_agentes_vivos())
        total_tesouros = sum(ag.tesouros for ag in self.agentes.values())
        total_explorado = len(self.conhecimento_global)
        
        return {
            "total_agentes": len(self.agentes),
            "agentes_vivos": agentes_vivos,
            "tesouros_coletados": total_tesouros,
            "celulas_exploradas": total_explorado,
            "por_modelo": self.estatisticas_modelos.copy()
        }
    
    def get_melhor_modelo(self):
        """
        Determina qual modelo ML teve melhor desempenho.
        
        Score = (tesouros × 10) - (mortes × 20)
        
        Returns:
            tuple: (nome_modelo, score)
        """
        self.atualizar_estatisticas()
        
        melhor_modelo = None
        melhor_score = -9999
        
        for modelo, stats in self.estatisticas_modelos.items():
            score = (stats['tesouros'] * 10) - (stats['mortes'] * 20)
            if score > melhor_score:
                melhor_score = score
                melhor_modelo = modelo
        
        return (melhor_modelo, melhor_score)
    
    def executar_turno(self):
        """
        Executa um turno de movimentação para todos os agentes vivos.
        
        Returns:
            list: Lista de resultados das ações
        """
        resultados = []
        for agente in self.get_agentes_vivos():
            proxima_celula = agente.escolher_proxima_celula()
            if proxima_celula:
                resultado = agente.explorar(*proxima_celula)
                resultados.append(resultado)
        
        # Sincronizar conhecimento após todos se moverem
        self.sincronizar_conhecimento()
        
        return resultados