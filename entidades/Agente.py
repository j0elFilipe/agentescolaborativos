import numpy as np

class Agente:
    def __init__(self, id, posicao, ambiente, modelo = None):
        self.id = id
        self.posicao = posicao #TUPLA (x,y)
        self.ambiente = ambiente
        self.tesouros = 0
        self.bombas_desa = 0
        self.vivo = True
        self.conhecimento_compartilhado = set() #CÉLULAS CONHECIDAS POR TODOS OS AGENTES
        self.historico_movimentos = []
        self.modelo = modelo #MODELO DE MACHINE LEARNING

    #FUNÇÃO ENCARREGUE DE REALIZAR A EXPLORAÇÃO
    def explorar(self, x, y):
        if not self.vivo:
            return {"status": "morto", "mensagem": f"Agente {self.id} está morto"}
        
        #VERIFICAR LIMITES DO AMBIENTE
        if not (0 <= x < self.ambiente.shape[0] and 0 <= y < self.ambiente.shape[1]):
            return {"status": "explorado", "celula": conteudo, "posicao": (x, y)}
        conteudo = self.ambiente[x, y]
        self.posicao = (x, y)
        self.historico_movimentos.append((x,y))

        resultado = {"status": "explorado", "celula": conteudo, "posicao": (x, y)}

        if conteudo == 'B':
            if self.bombas_desa > 0:
                self.bombas_desa = self.bombas_desa -  1
                self.ambiente[x, y] = 'E'
                resultado["mensagem"] = f"Agente {self.id} desactivou bomba em ({x}, {y})"
                resultado["accao"] = "desactivou_bomba"
            else:
                self.vivo = False
                self.ambiente[x, y] = 'E'
                resultado["mensagem"] = f"Agente {self.id} foi destruído em ({x}, {y})"
                resultado["accao"] = "destruido"
                resultado["status"] = "morto"
        
        elif conteudo == 'T':
            self.tesouros = self.tesouros + 1
            self.bombas_desa = self.bombas_desa + 1
            self.ambiente[x, y] = 'E'
            resultado["accao"] = f"Agente {self.id} encontrou tesouro em ({x}, {y})"
            resultado["accao"] = "tesouro"
        
        elif conteudo == 'F':
            resultado["mensagem"] = f"Agente {self.id} encontrou bandeira em ({x}, {y})"
            resultado["accao"] = "bandeira"

        elif conteudo == 'L':
            self.ambiente[x, y] = 'E'
            resultado["mensagem"] = f"Agente encontrou célula livre em ({x}, {y})"
            resultado["accao"] = "explorou"

        #ADD AO CONHECIMENTO COMPARTILHADO
        self.conhecimento_compartilhado.add((x, y))

        return resultado
    

    def escolher_proxima_celula(self):
        """
        ESCOLHER A PRÓXMA CÉLULA A EXPLORAR BASEADO NO MODELO DE ML.
        RETORNA (X, Y) OU NONE SE NÃO HOUVER CÉLULAS VÁLIDAS
        """

        x, y = self.posicao
        movimentos_possiveis = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # Adjacentes
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # Diagonais
        ]

        #FILTRAR MOVIMENTOS VÁLIDOS
        movimentos_validos = []
        for mx, my in movimentos_possiveis:
            if(0 <= mx < self.ambiente.shape[0] and
               0 <= my < self.ambiente.shape[1] and
               (mx, my) not in self.conhecimento_compartilhado):
                movimentos_validos.append((mx, my))
        
        if not movimentos_validos:
            #TENTAR QUALQUER CÉLULA NÃO EXPLORADA
            for i in range(self.ambiente.shape[0]):
                for j in range(self.ambiente.shape[1]):
                    if(i, j) not in self.conhecimento_compartilhado:
                        return (i, j)
            return None
        #SE TIVER MODELO, USAR PARA ESCOLHER O MELHOR MOVIMENTO
        if self.modelo:
            melhor_celula = self.modelo.prever_melhor_celula(movimentos_validos, self.posicao)
            return melhor_celula
        
        #CASO CONTRÁRIO, ESCOLHER ALEATORIAMENTE
        return movimentos_validos[np.random.randint(len(movimentos_validos))]
    
    
    def compartilhar_conhecimento(self, outros_agentes):
        """
        COMPARTILHA CONHECIMENTO COM OUTROS
        """
        for agente in outros_agentes:
            if agente.id != self.id:
                agente.conhecimento_compartilhado.update(self.conhecimento_compartilhado)
    
    def get_estado(self):
        """
        RETORNA O ESTADO ACTUAL DO AGENTE
        """
        return{
            "id": self.id,
            "posicao": self.posicao,
            "tesouros": self.tesouros,
            "bombas_desactivadas": self.bombas_desa,
            "vivo": self.vivo,
            "celulas_exploradas": len(self.conhecimento_compartilhado),
            "historico": len(self.historico_movimentos)
        }
    
    def __repr__(self):
        status = "Vivo" if self.vivo else "Morto"
        return f"Agente({self.id}, {status}, Pos: {self.posicao}, Tesouros: {self.tesouros})"


class GrupoAgentes:
    """
    FAZ A GESTÃO DE UM GRUPO DE AGENTES COLABORATIVOS
    """

    def __init__(self):
        self.agentes = {}
        self.conhecimento_global = set()


    def registrar_agente(self, agente):
        """
        REGISTA UM NOVO AGENTE NO GRUPO
        """
        if agente.id not in self.agentes:
            self.agentes[agente.id] = agente
            return True
    
        return False
    

    def sincronizar_conhecimento(self):
        """
        SINCRONIZA O CONHECIMENTO ENTRE TODOS OS AGENTES
        """
        #COLECTAR todo CONHECIMENTO
        for agente in self.agentes.values():
            self.conhecimento_global.update(agente.conhecimento_compartilhado)

        #DISTRIBUIR PARA TODOS
        for agente in self.agentes.values():
            agente.conhecimento_compartilhado = self.conhecimento_global.copy()
        
    
    def get_agentes_vivos(self):
        """
        RETORNA LISTA DE AGENTES VIVOS
        """
        return [ag for ag in self.agentes.values() if ag.vivo]
    
    
    def get_estatisticas(self):
        """
        RETORNA ESTATÍSTICAS DO GRUPO
        """
        agentes_vivos = len(self.get_agentes_vivos())
        total_tesouros = sum(ag.tesouros for ag in self.agentes.values())
        total_explorado = len(self.conhecimento_global)
        return{
            "total_agentes": len(self.agentes),
            "agentes_vivos": agentes_vivos,
            "tesouros_colectados": total_tesouros,
            "celulas_exploradas": total_explorado
        }
    

    def executar_turno(self):
        """
        EXECUTA UM TURNO DE MOVIMENTAÇÃO PARA TODOS OS AGENTES VIVOS
        """
        resultados = []
        for agente in self.get_agentes_vivos():
            proxima_celula = agente.escolher_proxima_celula()
            if  proxima_celula:
                resultado = agente.explorar(*proxima_celula)
                resultados.append(resultado)

        #SINCRONIZAR CONEHCIMENTO APÓS TODOS SE MOVEREM
        self.sincronizar_conhecimento()
        return resultados 


"""def mover_agente(agentes, ambiente, agente_id, x, y):
    if agente_id not in agentes:
        return {"erro": "Agente não registrado!"}

    agente = agentes[agente_id]
    if agente["status"] != "ativo":
        return {"erro": f"Agente {agente_id} está destruído e não pode se mover."}

    if not (0 <= x < 10 and 0 <= y < 10):
        return {"erro": "Movimento fora dos limites do ambiente!"}

    célula = ambiente[x, y]
    if célula == "T":
        agente["tesouros"] += 1
        ambiente[x, y] = "F"  # Tesouro foi coletado
        return {"mensagem": f"Agente {agente_id} encontrou um tesouro!"}
    elif célula == "B":
        agente["status"] = "destruído"
        return {"mensagem": f"Agente {agente_id} foi destruído por uma bomba!"}
    elif célula == "L":
        agente["posição"] = (x, y)
        return {"mensagem": f"Agente {agente_id} se moveu para uma célula livre."}
    else:
        return {"erro": "Ação não reconhecida!"}

def get_status_agente(agentes, agente_id):
    if agente_id in agentes:
        return agentes[agente_id]
    return {"erro": "Agente não encontrado!"}"""