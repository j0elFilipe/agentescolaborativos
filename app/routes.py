from flask import render_template, jsonify
from app import app
from entidades.Ambiente import criar_ambiente
from entidades.Agente import Agente

#INICIALIZANDO O AMBIENTE E AGENTES
ambiente = criar_ambiente()
agentes = [Agente(id = i, posicao = (0, 0), ambiente = ambiente) for i in range(2)]

@app.route("/")
def index():
    return render_template("interface.html")

@app.route("/actualizar", methods = ["GET"])
def actualizar():
    estado= {
        "ambiente": ambiente.tolist(),
        "agentes:": [{"id": a.id, "posicao": a.posicao, "vivo": a.vivo} for a in agentes]
    }

    return jsonify(estado)