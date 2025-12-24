class Agente:
    def __init__(self, id, posicao, ambiente):
        self.id = id
        self.posicao = posicao
        self.ambiente = ambiente
        self.tesouros = 0
        self.bombas_desa = 0
        self.vivo = True

#FUNÇÃO ENCARREGUE DE REALIZAR A EXPLORAÇÃO
    def explorar(self):
        if not self.vivo:
            return
        x, y = self.posicao
        conteudo = self.ambiente[x, y]
        if conteudo == 'B':
            if self.bombas_desa > 0:
                self.bombas_desa = self.bombas_desa -  1
            else:
                self.vivo = False
        
        elif conteudo == 'T':
            self.tesouros = self.tesouros + 1
            self.bombas_desa = self.bombas_desa + 1
        
        elif conteudo == 'F':
            print(f"Agente {self.id} encontrou a bandeira!")
        self.ambiente[x, y] = 'E' #MARCA A CÉLULA COMO EXPLORADA

    

    def registrar_agente(agentes, agente_id):
        if agente_id not in agentes:
            agentes[agente_id] = {
            "posição": (0, 0),  # Posição inicial (pode ser aleatória se preferir)
            "tesouros": 0,
            "status": "ativo",
            }
            return f"Agente com ID {agente_id} registrado com sucesso!"
    
        return f"Agente com ID {agente_id} já está registrado!"


def mover_agente(agentes, ambiente, agente_id, x, y):
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
    return {"erro": "Agente não encontrado!"}