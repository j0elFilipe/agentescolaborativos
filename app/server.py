from flask import Flask, jsonify, request
from entidades.Ambiente import criar_ambiente
from entidades.Agente import registrar_agente, mover_agente, get_status_agente
from app import app


# Inicializa o servidor Flask
app = Flask(__name__)

# Ambiente e agentes globais
ambiente = criar_ambiente()
agentes = {}
proximo_id = 1  # Contador global para os IDs dos agentes

# Rota para atualizar o estado do ambiente
@app.route("/actualizar", methods=["GET"])
def atualizar():
    return jsonify({"ambiente": ambiente.tolist()})

# Rota para registrar um novo agente
@app.route("/registrar", methods=["POST"])
def registrar():
    global proximo_id
    agente_id = proximo_id
    proximo_id += 1

    mensagem = registrar_agente(agentes, agente_id)
    return jsonify({"mensagem": mensagem, "id": agente_id})

# Rota para movimentar um agente
@app.route("/movimento", methods=["POST"])
def movimento():
    data = request.get_json()
    agente_id = data.get("id")
    x, y = data.get("x"), data.get("y")

    if agente_id is not None:
        resultado = mover_agente(agentes, ambiente, agente_id, x, y)
        return jsonify(resultado)
    return jsonify({"erro": "ID do agente não fornecido!"}), 400

# Rota para verificar o status de um agente
@app.route("/status/<int:agente_id>", methods=["GET"])
def status(agente_id):
    return jsonify(get_status_agente(agentes, agente_id))

# Função para iniciar o servidor
def iniciar_servidor():
    app.run(debug=True)