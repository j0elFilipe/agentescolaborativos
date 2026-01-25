import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import time
import threading
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class Agente:
    def __init__(self, id, x, y, modelo_tipo, modelo_ml=None):
        self.id = id
        self.x = x
        self.y = y
        self.tesouros = 0
        self.bombas_desativadas = 0
        self.vivo = True
        self.conhecimento = set()
        self.modelo_tipo = modelo_tipo
        self.modelo_ml = modelo_ml  # Modelo ML real treinado
        self.cor = self._gerar_cor(id)
        self.canvas_id = None  # ID do círculo no canvas
        self.rastro = []  # Histórico de posições
        self.movimentos = 0  # Contador de movimentos
        
    def _gerar_cor(self, id):
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                 '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8']
        return cores[id % len(cores)]

class Ambiente:
    def __init__(self, tamanho=10, perc_bombas=50):
        self.tamanho = tamanho
        self.perc_bombas = perc_bombas
        self.matriz = None
        self.tesouros_iniciais = 0
        self.criar_ambiente()
    
    def criar_ambiente(self):
        perc_livre = 100 - self.perc_bombas - 10
        self.matriz = np.random.choice(
            ['L', 'B', 'T'], 
            size=(self.tamanho, self.tamanho),
            p=[perc_livre/100, self.perc_bombas/100, 0.10]
        )
        fx, fy = np.random.randint(0, self.tamanho, 2)
        self.matriz[fx, fy] = 'F'
        self.tesouros_iniciais = np.sum(self.matriz == 'T')

class ModeloML:
    def __init__(self, tipo='knn'):
        self.tipo = tipo
        if tipo == 'knn':
            self.modelo = KNeighborsClassifier(n_neighbors=3)
        elif tipo == 'tree':
            self.modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif tipo == 'bayes':
            self.modelo = GaussianNB()
        self.treinar_modelo_base()
    
    def treinar_modelo_base(self):
        X = []
        y = []
        # Gerar dados de treino mais realistas
        for _ in range(2000):
            x, y_coord = np.random.randint(0, 10, 2)
            distancia_centro = np.sqrt((x-5)**2 + (y_coord-5)**2)
            distancia_borda = min(x, y_coord, 9-x, 9-y_coord)
            
            # Lógica: bombas no centro, tesouros nas bordas
            rand = np.random.random()
            if distancia_centro < 3:
                label = 'B' if rand < 0.5 else ('L' if rand < 0.8 else 'T')
            elif distancia_borda <= 1:
                label = 'T' if rand < 0.4 else ('L' if rand < 0.7 else 'B')
            else:
                label = 'L' if rand < 0.6 else ('B' if rand < 0.8 else 'T')
            
            X.append([x, y_coord, distancia_centro, distancia_borda])
            y.append(label)
        
        self.modelo.fit(X, y)
    
    def escolher_melhor_celula(self, celulas_possiveis):
        """Escolhe a melhor célula usando o modelo ML"""
        if not celulas_possiveis:
            return None
        
        melhor_celula = None
        melhor_score = -1000
        
        for cx, cy in celulas_possiveis:
            dist_centro = np.sqrt((cx-5)**2 + (cy-5)**2)
            dist_borda = min(cx, cy, 9-cx, 9-cy)
            
            # Prever tipo de célula
            predicao = self.modelo.predict([[cx, cy, dist_centro, dist_borda]])[0]
            
            # Sistema de pontuação
            score = 0
            if predicao == 'T':  # Tesouro é o melhor
                score = 100
            elif predicao == 'L':  # Livre é neutro
                score = 50
            elif predicao == 'B':  # Bomba é ruim
                score = -50
            
            # Adicionar aleatoriedade para exploração
            score += np.random.randint(-10, 10)
            
            if score > melhor_score:
                melhor_score = score
                melhor_celula = (cx, cy)
        
        return melhor_celula if melhor_celula else celulas_possiveis[0]

class SistemaAgentesColaborativos:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Agentes Colaborativos - IA 2024/2025")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.ambiente = None
        self.agentes = []
        self.executando = False
        self.pausado = False
        self.abordagem = 'A'
        self.num_agentes = 2
        self.perc_bombas = 50
        self.logs = []
        self.tempo_inicio = 0
        self.velocidade = 500  # Milissegundos entre movimentos
        self.mostrar_rastros = True  # Mostrar trilhas dos agentes
        
        self.tamanho_celula = 45  # Reduzido de 50 para 45
        self.criar_interface()
    
    def criar_interface(self):
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        titulo = tk.Label(main_frame, text="🤖 Sistema de Agentes Colaborativos", 
                         font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        titulo.pack(pady=10)
        
        # Frame superior
        top_frame = tk.Frame(main_frame, bg='#f0f0f0')
        top_frame.pack(fill=tk.X, pady=10)
        
        # Painel de Controle
        control_frame = tk.LabelFrame(top_frame, text="Configurações", 
                                      font=('Arial', 12, 'bold'), bg='white', padx=15, pady=15)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Abordagens
        tk.Label(control_frame, text="Abordagem:", font=('Arial', 10, 'bold'), 
                bg='white').grid(row=0, column=0, sticky='w', pady=5)
        
        abordagem_frame = tk.Frame(control_frame, bg='white')
        abordagem_frame.grid(row=0, column=1, columnspan=3, pady=5)
        
        self.abordagem_var = tk.StringVar(value='A')
        for i, (ab, desc) in enumerate([('A', '>50% tesouros'), 
                                        ('B', 'Explorar tudo'), 
                                        ('C', 'Encontrar bandeira')]):
            btn = tk.Radiobutton(abordagem_frame, text=f"{ab}", variable=self.abordagem_var,
                                value=ab, font=('Arial', 10, 'bold'), bg='white',
                                selectcolor='#3498db', activebackground='white',
                                command=lambda a=ab: setattr(self, 'abordagem', a))
            btn.pack(side=tk.LEFT, padx=5)
            tk.Label(abordagem_frame, text=f"({desc})", font=('Arial', 8), 
                    bg='white', fg='#7f8c8d').pack(side=tk.LEFT, padx=5)
        
        # Número de agentes
        tk.Label(control_frame, text="Nº Agentes:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=1, column=0, sticky='w', pady=5)
        self.agentes_scale = tk.Scale(control_frame, from_=2, to=10, orient=tk.HORIZONTAL,
                                     bg='white', length=200, command=self.atualizar_num_agentes)
        self.agentes_scale.set(2)
        self.agentes_scale.grid(row=1, column=1, columnspan=2, pady=5)
        self.agentes_label = tk.Label(control_frame, text="2", font=('Arial', 10, 'bold'),
                                      bg='white', fg='#3498db')
        self.agentes_label.grid(row=1, column=3, pady=5)
        
        # Percentual de bombas
        tk.Label(control_frame, text="% Bombas:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=2, column=0, sticky='w', pady=5)
        self.bombas_scale = tk.Scale(control_frame, from_=20, to=80, orient=tk.HORIZONTAL,
                                     bg='white', length=200, command=self.atualizar_bombas)
        self.bombas_scale.set(50)
        self.bombas_scale.grid(row=2, column=1, columnspan=2, pady=5)
        self.bombas_label = tk.Label(control_frame, text="50%", font=('Arial', 10, 'bold'),
                                     bg='white', fg='#e74c3c')
        self.bombas_label.grid(row=2, column=3, pady=5)
        
        # Velocidade
        tk.Label(control_frame, text="Velocidade:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=3, column=0, sticky='w', pady=5)
        self.velocidade_scale = tk.Scale(control_frame, from_=100, to=1000, orient=tk.HORIZONTAL,
                                         bg='white', length=200, command=self.atualizar_velocidade,
                                         resolution=100)
        self.velocidade_scale.set(500)
        self.velocidade_scale.grid(row=3, column=1, columnspan=2, pady=5)
        self.velocidade_label = tk.Label(control_frame, text="500ms", font=('Arial', 10, 'bold'),
                                         bg='white', fg='#9b59b6')
        self.velocidade_label.grid(row=3, column=3, pady=5)
        
        # Checkbox para mostrar rastros
        self.mostrar_rastros_var = tk.BooleanVar(value=True)
        rastros_check = tk.Checkbutton(control_frame, text="Mostrar Rastros dos Agentes", 
                                       variable=self.mostrar_rastros_var,
                                       font=('Arial', 10), bg='white',
                                       command=self.toggle_rastros)
        rastros_check.grid(row=4, column=0, columnspan=4, pady=5, sticky='w')
        
        # Botões de controle
        btn_frame = tk.Frame(control_frame, bg='white')
        btn_frame.grid(row=5, column=0, columnspan=4, pady=15)
        
        self.btn_iniciar = tk.Button(btn_frame, text="▶ Iniciar", font=('Arial', 11, 'bold'),
                                     bg='#27ae60', fg='white', width=10, height=2,
                                     command=self.iniciar_simulacao)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        self.btn_pausar = tk.Button(btn_frame, text="⏸ Pausar", font=('Arial', 11, 'bold'),
                                    bg='#f39c12', fg='white', width=10, height=2,
                                    command=self.pausar_simulacao, state=tk.DISABLED)
        self.btn_pausar.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = tk.Button(btn_frame, text="↻ Reset", font=('Arial', 11, 'bold'),
                                   bg='#e74c3c', fg='white', width=10, height=2,
                                   command=self.resetar_simulacao)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Painel de Métricas
        metrics_frame = tk.LabelFrame(top_frame, text="Métricas em Tempo Real",
                                     font=('Arial', 12, 'bold'), bg='white', padx=15, pady=15)
        metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        metrics_data = [
            ("👥 Agentes Vivos:", 'agentes_vivos', '#3498db'),
            ("💎 Tesouros:", 'tesouros', '#f1c40f'),
            ("🎯 Células Exploradas:", 'exploradas', '#2ecc71'),
            ("⏱ Tempo (s):", 'tempo', '#9b59b6')
        ]
        
        self.metric_labels = {}
        for i, (nome, key, cor) in enumerate(metrics_data):
            frame = tk.Frame(metrics_frame, bg=cor, padx=10, pady=8)
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=nome, font=('Arial', 10, 'bold'), 
                    bg=cor, fg='white').pack(side=tk.LEFT)
            label = tk.Label(frame, text="0", font=('Arial', 14, 'bold'), 
                           bg=cor, fg='white')
            label.pack(side=tk.RIGHT)
            self.metric_labels[key] = label
        
        # Frame para estatísticas dos modelos ML
        stats_ml_frame = tk.LabelFrame(top_frame, text="Performance dos Modelos ML",
                                       font=('Arial', 10, 'bold'), bg='white', padx=10, pady=10)
        stats_ml_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.stats_ml_labels = {}
        for modelo in ['knn', 'tree', 'bayes']:
            frame = tk.Frame(stats_ml_frame, bg='#ecf0f1', padx=8, pady=6, relief=tk.RIDGE, borderwidth=2)
            frame.pack(fill=tk.X, pady=3)
            
            nome_label = tk.Label(frame, text=f"{modelo.upper()}:", font=('Arial', 9, 'bold'),
                                 bg='#ecf0f1', fg='#2c3e50')
            nome_label.pack(anchor='w')
            
            info_label = tk.Label(frame, text="Tesouros: 0 | Mortes: 0", font=('Arial', 8),
                                 bg='#ecf0f1', fg='#34495e')
            info_label.pack(anchor='w')
            
            self.stats_ml_labels[modelo] = info_label
        
        # Frame do meio
        middle_frame = tk.Frame(main_frame, bg='#f0f0f0')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas do Ambiente
        ambiente_frame = tk.LabelFrame(middle_frame, text="Ambiente 10x10",
                                      font=('Arial', 12, 'bold'), bg='white', padx=5, pady=5)
        ambiente_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas com tamanho ajustado (10 células x 45px = 450px + margem)
        canvas_container = tk.Frame(ambiente_frame, bg='white')
        canvas_container.pack(expand=True)
        
        self.canvas = tk.Canvas(canvas_container, width=450, height=450, bg='white', 
                               highlightthickness=1, highlightbackground='#bdc3c7')
        self.canvas.pack()
        
        # Painel de Logs
        logs_frame = tk.LabelFrame(middle_frame, text="Logs do Sistema",
                                  font=('Arial', 12, 'bold'), bg='white', padx=10, pady=10)
        logs_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, width=50, height=30,
                                                  font=('Courier', 9), bg='#2c3e50', 
                                                  fg='#ecf0f1', wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Legenda
        legenda_frame = tk.Frame(main_frame, bg='white', padx=10, pady=10)
        legenda_frame.pack(fill=tk.X)
        
        tk.Label(legenda_frame, text="Legenda:", font=('Arial', 10, 'bold'),
                bg='white').pack(side=tk.LEFT, padx=10)
        
        legendas = [
            ("🟢 Livre", '#c8e6c9'),
            ("🔴 Bomba", '#ffcdd2'),
            ("💎 Tesouro", '#fff9c4'),
            ("🏁 Bandeira", '#e1bee7'),
            ("⬜ Explorada", '#e0e0e0')
        ]
        
        for texto, cor in legendas:
            frame = tk.Frame(legenda_frame, bg=cor, padx=8, pady=4, relief=tk.RAISED, borderwidth=1)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=texto, bg=cor, font=('Arial', 9)).pack()
    
    def atualizar_num_agentes(self, val):
        self.num_agentes = int(val)
        self.agentes_label.config(text=str(val))
    
    def atualizar_bombas(self, val):
        self.perc_bombas = int(val)
        self.bombas_label.config(text=f"{val}%")
    
    def atualizar_velocidade(self, val):
        self.velocidade = int(val)
        self.velocidade_label.config(text=f"{val}ms")
    
    def toggle_rastros(self):
        self.mostrar_rastros = self.mostrar_rastros_var.get()
        if not self.mostrar_rastros:
            self.canvas.delete('rastro')  # Limpa rastros existentes
    
    def adicionar_log(self, mensagem):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {mensagem}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.logs.append(log_entry)
    
    def desenhar_grid_ambiente(self):
        """Desenha o grid do ambiente (células de fundo)"""
        cores_celulas = {
            'L': '#c8e6c9',
            'B': '#ffcdd2',
            'T': '#fff9c4',
            'F': '#e1bee7',
            'E': '#e0e0e0'
        }
        
        for i in range(10):
            for j in range(10):
                x1 = j * self.tamanho_celula
                y1 = i * self.tamanho_celula
                x2 = x1 + self.tamanho_celula
                y2 = y1 + self.tamanho_celula
                
                celula = self.ambiente.matriz[i, j]
                cor = cores_celulas.get(celula, 'white')
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=cor, outline='#95a5a6', 
                                            width=1, tags='grid')
                
                # Símbolos (tamanho ajustado para 45px)
                cx = x1 + self.tamanho_celula/2
                cy = y1 + self.tamanho_celula/2
                
                if celula == 'B':
                    self.canvas.create_text(cx, cy, text="💣", font=('Arial', 14), tags='grid')
                elif celula == 'T':
                    self.canvas.create_text(cx, cy, text="💎", font=('Arial', 14), tags='grid')
                elif celula == 'F':
                    self.canvas.create_text(cx, cy, text="🏁", font=('Arial', 14), tags='grid')
    
    def criar_agente_visual(self, agente):
        """Cria a representação visual de um agente no canvas"""
        x = agente.y * self.tamanho_celula + self.tamanho_celula/2
        y = agente.x * self.tamanho_celula + self.tamanho_celula/2
        
        # Adicionar posição inicial ao rastro
        agente.rastro.append((agente.x, agente.y))
        
        # Criar círculo do agente (ajustado para células de 45px)
        raio = 13  # Raio do círculo do agente
        circulo = self.canvas.create_oval(
            x-raio, y-raio, x+raio, y+raio, 
            fill=agente.cor, 
            outline='yellow', 
            width=2,
            tags=f'agente_{agente.id}'
        )
        
        # Criar texto com ID do agente
        texto = self.canvas.create_text(
            x, y, 
            text=str(agente.id), 
            font=('Arial', 10, 'bold'), 
            fill='white',
            tags=f'agente_{agente.id}'
        )
        
        # Criar indicador de modelo ML (menor e mais próximo)
        modelo_texto = self.canvas.create_text(
            x, y + 20,
            text=agente.modelo_tipo.upper(),
            font=('Arial', 5, 'bold'),
            fill=agente.cor,
            tags=f'agente_{agente.id}'
        )
        
        agente.canvas_id = f'agente_{agente.id}'
        
        return circulo
    
    def mover_agente_visual(self, agente, nova_x, nova_y):
        """Move visualmente o agente no canvas com animação"""
        if not agente.vivo:
            return
        
        # Posição atual no canvas
        x_atual = agente.y * self.tamanho_celula + self.tamanho_celula/2
        y_atual = agente.x * self.tamanho_celula + self.tamanho_celula/2
        
        # Nova posição no canvas
        x_novo = nova_y * self.tamanho_celula + self.tamanho_celula/2
        y_novo = nova_x * self.tamanho_celula + self.tamanho_celula/2
        
        # Calcular deslocamento
        dx = x_novo - x_atual
        dy = y_novo - y_atual
        
        # Mover todos os elementos do agente
        self.canvas.move(agente.canvas_id, dx, dy)
        
        # Atualizar posição lógica
        agente.x = nova_x
        agente.y = nova_y
    
    def remover_agente_visual(self, agente):
        """Remove o agente do canvas quando morre"""
        # Criar efeito de explosão (ajustado para 45px)
        x = agente.y * self.tamanho_celula + self.tamanho_celula/2
        y = agente.x * self.tamanho_celula + self.tamanho_celula/2
        
        # Estrela de explosão
        for i in range(8):
            angle = i * 45
            dx = 15 * np.cos(np.radians(angle))
            dy = 15 * np.sin(np.radians(angle))
            self.canvas.create_line(x, y, x+dx, y+dy, fill='red', width=2, tags='explosao')
        
        self.canvas.create_text(x, y-25, text="💥", font=('Arial', 24), tags='explosao')
        
        # Remover explosão após 500ms
        self.root.after(500, lambda: self.canvas.delete('explosao'))
        
        self.canvas.delete(agente.canvas_id)
    
    def desenhar_rastro(self, agente, x_ant, y_ant, x_novo, y_novo):
        """Desenha uma linha mostrando o caminho percorrido"""
        x1 = y_ant * self.tamanho_celula + self.tamanho_celula/2
        y1 = x_ant * self.tamanho_celula + self.tamanho_celula/2
        x2 = y_novo * self.tamanho_celula + self.tamanho_celula/2
        y2 = x_novo * self.tamanho_celula + self.tamanho_celula/2
        
        # Linha pontilhada com cor do agente (mais transparente)
        self.canvas.create_line(x1, y1, x2, y2, 
                               fill=agente.cor, 
                               width=2, 
                               dash=(4, 4),
                               tags='rastro')
    
    def desenhar_seta_direcao(self, agente, x_atual, y_atual, dx, dy):
        """Desenha uma seta mostrando a direção do movimento"""
        # Normalizar direção
        tamanho_seta = 25
        if abs(dx) > 0 or abs(dy) > 0:
            comp = np.sqrt(dx**2 + dy**2)
            dx_norm = (dx / comp) * tamanho_seta
            dy_norm = (dy / comp) * tamanho_seta
            
            # Criar seta
            seta = self.canvas.create_line(
                x_atual, y_atual,
                x_atual + dx_norm, y_atual + dy_norm,
                arrow=tk.LAST,
                fill=agente.cor,
                width=3,
                tags='seta_dir'
            )
            
            # Remover seta após movimento
            self.root.after(self.velocidade - 50, lambda: self.canvas.delete('seta_dir'))
    
    def animar_movimento(self, agente):
        """Anima a borda do agente ao se mover"""
        # Encontrar o círculo do agente
        items = self.canvas.find_withtag(agente.canvas_id)
        if items:
            circulo = items[0]
            # Alterar borda temporariamente
            self.canvas.itemconfig(circulo, outline='lime', width=5)
            # Voltar ao normal
            self.root.after(200, lambda: self.canvas.itemconfig(circulo, outline='yellow', width=4))
    
    def iniciar_simulacao(self):
        try:
            # DEBUG: Verificar se atributo existe
            if not hasattr(self, 'estatisticas_modelos'):
                self.adicionar_log("⚠️ CRIANDO estatisticas_modelos (não existia!)")
                self.estatisticas_modelos = {
                    'knn': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []},
                    'tree': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []},
                    'bayes': {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []}
                }
            
            if self.pausado:
                # Retomar simulação pausada
                self.pausado = False
                self.executando = True
                self.btn_iniciar.config(state=tk.DISABLED)
                self.btn_pausar.config(state=tk.NORMAL, text="⏸ Pausar")
                self.adicionar_log("▶️ Simulação retomada")
                self.loop_simulacao()
                return
            
            # Nova simulação
            self.resetar_simulacao()
            self.executando = True
            self.btn_iniciar.config(state=tk.DISABLED)
            self.btn_pausar.config(state=tk.NORMAL)
            self.agentes_scale.config(state=tk.DISABLED)
            self.bombas_scale.config(state=tk.DISABLED)
            
            # Criar ambiente
            self.ambiente = Ambiente(tamanho=10, perc_bombas=self.perc_bombas)
            self.adicionar_log(f"🚀 Simulação iniciada - Abordagem {self.abordagem}")
            
            # Treinar modelos ML
            self.adicionar_log("🧠 Treinando modelos de Machine Learning...")
            self.modelos_ml = {
                'knn': ModeloML('knn'),
                'tree': ModeloML('tree'),
                'bayes': ModeloML('bayes')
            }
            self.adicionar_log("✅ Modelos KNN, Tree e Bayes treinados!")
            
            # Resetar estatísticas
            for modelo in self.estatisticas_modelos:
                self.estatisticas_modelos[modelo] = {'tesouros': 0, 'mortes': 0, 'movimentos': 0, 'agentes': []}
            
            # Desenhar grid
            self.desenhar_grid_ambiente()
            
            # Criar agentes com modelos ML
            modelos_tipos = ['knn', 'tree', 'bayes']
            self.agentes = []
            for i in range(self.num_agentes):
                x, y = np.random.randint(0, 10, 2)
                while self.ambiente.matriz[x, y] == 'B':
                    x, y = np.random.randint(0, 10, 2)
                
                modelo_tipo = modelos_tipos[i % len(modelos_tipos)]
                modelo_ml = self.modelos_ml[modelo_tipo]  # Passa o modelo treinado
                
                # CORRIGIDO: Criar agente com x, y separados (sem ambiente como parâmetro na classe interna)
                agente = Agente(i, x, y, modelo_tipo, modelo_ml)
                self.agentes.append(agente)
                
                # Registrar agente nas estatísticas
                self.estatisticas_modelos[modelo_tipo]['agentes'].append(i)
                
                # Criar representação visual
                self.criar_agente_visual(agente)
                
                self.adicionar_log(f"👤 Agente {i} criado ({modelo_tipo.upper()}) em ({x},{y})")
            
            self.adicionar_log(f"💣 {self.perc_bombas}% de bombas")
            self.adicionar_log(f"💎 {self.ambiente.tesouros_iniciais} tesouros disponíveis")
            
            self.tempo_inicio = time.time()
            
            # Iniciar loop de simulação
            self.loop_simulacao()
            
        except Exception as e:
            self.adicionar_log(f"❌ ERRO ao iniciar simulação: {str(e)}")
            import traceback
            self.adicionar_log(f"Detalhes: {traceback.format_exc()}")
            self.executando = False
            self.btn_iniciar.config(state=tk.NORMAL)
            self.btn_pausar.config(state=tk.DISABLED)
    
    def loop_simulacao(self):
        """Loop principal da simulação"""
        try:
            if not self.executando or self.pausado:
                return
            
            agentes_vivos = [ag for ag in self.agentes if ag.vivo]
            
            if not agentes_vivos:
                self.executando = False
                self.adicionar_log("❌ Todos os agentes foram destruídos!")
                self.finalizar_simulacao()
                return
            
            # Mover cada agente vivo
            for agente in agentes_vivos:
                self.executar_movimento_agente(agente)
            
            # Atualizar ambiente visual
            self.atualizar_ambiente_visual()
            
            # Atualizar métricas
            self.atualizar_metricas()
            
            # Verificar condição de sucesso
            if self.verificar_sucesso():
                self.finalizar_simulacao()
                return
            
            # Agendar próxima iteração
            self.root.after(self.velocidade, self.loop_simulacao)
            
        except Exception as e:
            self.adicionar_log(f"❌ ERRO no loop: {str(e)}")
            import traceback
            self.adicionar_log(f"Detalhes: {traceback.format_exc()}")
            self.executando = False
    
    def executar_movimento_agente(self, agente):
        """Executa um movimento de um agente USANDO O MODELO ML"""
        movimentos = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        # Filtrar movimentos possíveis
        possiveis = []
        for dx, dy in movimentos:
            nx, ny = agente.x + dx, agente.y + dy
            if (0 <= nx < 10 and 0 <= ny < 10 and 
                f"{nx},{ny}" not in agente.conhecimento):
                possiveis.append((nx, ny))
        
        if not possiveis:
            return
        
        # ========== USAR MODELO ML PARA ESCOLHER ========== #
        if agente.modelo_ml:
            nx, ny = agente.modelo_ml.escolher_melhor_celula(possiveis)
        else:
            # Fallback aleatório (não deveria acontecer)
            nx, ny = possiveis[np.random.randint(len(possiveis))]
        # ================================================== #
        
        celula = self.ambiente.matriz[nx, ny]
        
        # Incrementar contador de movimentos do modelo
        self.estatisticas_modelos[agente.modelo_tipo]['movimentos'] += 1
        agente.movimentos += 1
        
        # Mover visualmente
        self.mover_agente_visual(agente, nx, ny)
        
        # Compartilhar conhecimento
        for ag in self.agentes:
            ag.conhecimento.add(f"{nx},{ny}")
        
        # Processar célula
        if celula == 'B':
            if agente.bombas_desativadas > 0:
                agente.bombas_desativadas -= 1
                self.adicionar_log(f"🛡️ Agente {agente.id} ({agente.modelo_tipo.upper()}) desativou bomba ({nx},{ny})")
                self.ambiente.matriz[nx, ny] = 'E'
                self.efeito_desativacao(nx, ny)
            else:
                agente.vivo = False
                self.estatisticas_modelos[agente.modelo_tipo]['mortes'] += 1
                self.adicionar_log(f"💥 Agente {agente.id} ({agente.modelo_tipo.upper()}) DESTRUÍDO em ({nx},{ny})")
                self.ambiente.matriz[nx, ny] = 'E'
                self.remover_agente_visual(agente)
        
        elif celula == 'T':
            agente.tesouros += 1
            agente.bombas_desativadas += 1
            self.estatisticas_modelos[agente.modelo_tipo]['tesouros'] += 1
            self.adicionar_log(f"💎 Agente {agente.id} ({agente.modelo_tipo.upper()}) achou TESOURO ({nx},{ny}) [Total: {agente.tesouros}]")
            self.ambiente.matriz[nx, ny] = 'E'
            self.efeito_tesouro(nx, ny)
        
        elif celula == 'F':
            self.adicionar_log(f"🏁 Agente {agente.id} ({agente.modelo_tipo.upper()}) achou BANDEIRA ({nx},{ny})!")
            self.efeito_bandeira(nx, ny)
        
        elif celula == 'L':
            self.ambiente.matriz[nx, ny] = 'E'
        
        # Atualizar estatísticas ML
        self.atualizar_estatisticas_ml()
    
    def efeito_tesouro(self, x, y):
        """Efeito visual ao coletar tesouro"""
        cx = y * self.tamanho_celula + self.tamanho_celula/2
        cy = x * self.tamanho_celula + self.tamanho_celula/2
        
        # Estrelas brilhantes (ajustadas para células de 45px)
        for i in range(4):
            self.canvas.create_text(
                cx + np.random.randint(-10, 10),
                cy + np.random.randint(-10, 10),
                text="✨",
                font=('Arial', 12),
                tags='efeito_tesouro'
            )
        
        self.root.after(400, lambda: self.canvas.delete('efeito_tesouro'))
    
    def efeito_desativacao(self, x, y):
        """Efeito visual ao desativar bomba"""
        cx = y * self.tamanho_celula + self.tamanho_celula/2
        cy = x * self.tamanho_celula + self.tamanho_celula/2
        
        self.canvas.create_text(cx, cy, text="🛡️", font=('Arial', 18), tags='efeito_escudo')
        self.root.after(300, lambda: self.canvas.delete('efeito_escudo'))
    
    def efeito_bandeira(self, x, y):
        """Efeito visual ao encontrar bandeira"""
        cx = y * self.tamanho_celula + self.tamanho_celula/2
        cy = x * self.tamanho_celula + self.tamanho_celula/2
        
        # Círculo pulsante (ajustado para 45px)
        for r in range(6, 25, 6):
            circulo = self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r,
                outline='purple',
                width=2,
                tags='efeito_bandeira'
            )
        
        self.root.after(500, lambda: self.canvas.delete('efeito_bandeira'))
    
    def atualizar_ambiente_visual(self):
        """Atualiza apenas as células que mudaram"""
        self.canvas.delete('grid')
        self.desenhar_grid_ambiente()
        
        # Redesenhar agentes por cima
        for agente in self.agentes:
            if agente.vivo:
                self.canvas.tag_raise(agente.canvas_id)
    
    def verificar_sucesso(self):
        tesouros_encontrados = sum(ag.tesouros for ag in self.agentes)
        tesouros_restantes = np.sum(self.ambiente.matriz == 'T')
        total_tesouros = tesouros_encontrados + tesouros_restantes
        agentes_vivos = sum(1 for ag in self.agentes if ag.vivo)
        
        sucesso = False
        
        if self.abordagem == 'A':
            if total_tesouros > 0:
                perc = (tesouros_encontrados / total_tesouros) * 100
                if perc > 50:
                    sucesso = True
                    self.adicionar_log(f"✅ SUCESSO! {perc:.1f}% dos tesouros encontrados!")
        
        elif self.abordagem == 'B':
            todas_exploradas = not np.any(np.isin(self.ambiente.matriz, ['L', 'B', 'T']))
            if todas_exploradas and agentes_vivos > 0:
                sucesso = True
                self.adicionar_log(f"✅ SUCESSO! Ambiente explorado com {agentes_vivos} agente(s)!")
        
        elif self.abordagem == 'C':
            for ag in self.agentes:
                if ag.vivo and self.ambiente.matriz[ag.x, ag.y] == 'F':
                    sucesso = True
                    self.adicionar_log(f"✅ SUCESSO! Agente {ag.id} encontrou a bandeira!")
                    break
        
        if sucesso or agentes_vivos == 0:
            self.executando = False
            return True
        
        return False
    
    def atualizar_metricas(self):
        agentes_vivos = sum(1 for ag in self.agentes if ag.vivo)
        tesouros = sum(ag.tesouros for ag in self.agentes)
        exploradas = np.sum(self.ambiente.matriz == 'E')
        tempo = time.time() - self.tempo_inicio
        
        self.metric_labels['agentes_vivos'].config(text=str(agentes_vivos))
        self.metric_labels['tesouros'].config(text=str(tesouros))
        self.metric_labels['exploradas'].config(text=str(exploradas))
        self.metric_labels['tempo'].config(text=f"{tempo:.1f}")
    
    def atualizar_estatisticas_ml(self):
        """Atualiza as estatísticas de performance dos modelos ML"""
        # Verificar se stats_ml_labels existe
        if not hasattr(self, 'stats_ml_labels'):
            return
        
        for modelo in ['knn', 'tree', 'bayes']:
            stats = self.estatisticas_modelos[modelo]
            texto = f"💎 Tesouros: {stats['tesouros']} | 💀 Mortes: {stats['mortes']}"
            if modelo in self.stats_ml_labels:
                self.stats_ml_labels[modelo].config(text=texto)
    
    def pausar_simulacao(self):
        if self.executando:
            self.pausado = True
            self.executando = False
            self.btn_pausar.config(text="▶️ Continuar")
            self.btn_iniciar.config(state=tk.NORMAL)
            self.adicionar_log("⏸ Simulação PAUSADA")
        else:
            self.pausado = False
            self.executando = True
            self.btn_pausar.config(text="⏸ Pausar")
            self.btn_iniciar.config(state=tk.DISABLED)
            self.adicionar_log("▶️ Simulação RETOMADA")
            self.loop_simulacao()
    
    def resetar_simulacao(self):
        self.executando = False
        self.pausado = False
        self.agentes = []
        self.logs = []
        self.tempo_inicio = 0
        self.canvas.delete("all")  # Apaga tudo incluindo rastros
        self.log_text.delete(1.0, tk.END)
        
        for label in self.metric_labels.values():
            label.config(text="0")
        
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_pausar.config(state=tk.DISABLED, text="⏸ Pausar")
        self.agentes_scale.config(state=tk.NORMAL)
        self.bombas_scale.config(state=tk.NORMAL)
    
    def finalizar_simulacao(self):
        tempo_total = time.time() - self.tempo_inicio
        self.adicionar_log(f"🏁 Simulação FINALIZADA em {tempo_total:.2f}s")
        
        # Mostrar comparação final dos modelos
        self.adicionar_log("\n📊 === COMPARAÇÃO DOS MODELOS ML ===")
        melhor_modelo = None
        melhor_score = -1000
        
        for modelo in ['knn', 'tree', 'bayes']:
            stats = self.estatisticas_modelos[modelo]
            # Score: tesouros valem +10, mortes valem -20
            score = (stats['tesouros'] * 10) - (stats['mortes'] * 20)
            taxa_sucesso = (stats['tesouros'] / max(stats['movimentos'], 1)) * 100
            
            self.adicionar_log(f"{modelo.upper()}: {stats['tesouros']} tesouros, {stats['mortes']} mortes, Score: {score}")
            
            if score > melhor_score:
                melhor_score = score
                melhor_modelo = modelo
        
        self.adicionar_log(f"🏆 MELHOR MODELO: {melhor_modelo.upper()} (Score: {melhor_score})")
        
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_pausar.config(state=tk.DISABLED)
        self.agentes_scale.config(state=tk.NORMAL)
        self.bombas_scale.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaAgentesColaborativos(root)
    root.mainloop()