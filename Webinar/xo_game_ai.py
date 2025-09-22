"""
🧠 X și O cu Inteligență Artificială prin Machine Learning (Q-Learning)
================================================================================

Acest script demonstrează diferența dintre:
1. AI simplu cu reguli fixe (cum am văzut în xo_game.py)
2. AI cu Machine Learning care "învață" prin experiență

🎯 CONCEPTE EDUCAȚIONALE CHEIE:
- Q-Learning: Algoritmul învață prin încercare și eroare
- Exploration vs Exploitation: Explorează strategii noi vs folosește ce știe
- Reward System: Învață din recompense (+1 câștig, -1 pierdere)
- Q-Table: "Memoria" AI-ului - ce să facă în fiecare situație

🔍 DIFERENȚA FUNDAMENTALĂ:
- AI simplu: "Dacă pot câștiga → câștig, dacă trebuie să blochez → blochez"
- AI cu ML: "Am încercat această mutare înainte și am câștigat/pierdut,
            deci o să o încerc din nou/evit"

Autori: Neo & Claude - Webinar "Arta Programării"
"""

import sys
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QDialog, QTextEdit, QFrame,
                            QScrollArea, QGridLayout, QSplitter, QProgressBar,
                            QTabWidget, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


class QLearningAgent:
    """
    🧠 AGENTUL Q-LEARNING - "Creierul" AI-ului care învață

    Concepte cheie:
    - Q-Table: Dicționar cu situații de joc și ce să facă
    - Epsilon (ε): Cât de mult explorează vs folosește ce știe
    - Learning Rate (α): Cât de repede învață din experiențe noi
    - Discount Factor (γ): Cât de mult valorează recompensele viitoare
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # 📚 Q-Table: Memoria AI-ului - pentru fiecare (situație, acțiune)
        # păstrează o valoare Q care spune cât de bună e acea acțiune
        self.q_table = defaultdict(float)

        # 🎓 Parametrii de învățare
        self.learning_rate = learning_rate      # α - cât de repede învață (0.1 = încet dar sigur)
        self.discount_factor = discount_factor  # γ - importanța viitorului (0.95 = se gândește la viitor)
        self.epsilon = epsilon                  # ε - cât explorează (0.1 = 10% explorare, 90% folosește ce știe)

        # 📊 Statistici pentru monitorizare
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # 🎯 Pentru debugging și educație
        self.last_state = None
        self.last_action = None
        self.decision_reason = ""

    def state_to_string(self, board):
        """
        🔄 Convertește tabla de joc într-un string pentru Q-table

        De ce string? Pentru că Q-table-ul are nevoie de chei immutable
        Exemplu: ['X', '', 'O', '', 'X', '', '', '', ''] → "X_O_X____"
        """
        return ''.join(board).replace(' ', '_')

    def get_valid_actions(self, board):
        """
        🎯 Găsește toate mișcările posibile (pozițiile libere)
        """
        return [i for i, cell in enumerate(board) if cell == '']

    def choose_action(self, board, training=True):
        """
        🤔 DECIZIA PRINCIPALĂ: Ce mișcare să fac?

        Folosește strategia ε-greedy:
        - Cu probabilitatea ε: EXPLOREAZĂ (încearcă ceva nou/random)
        - Cu probabilitatea 1-ε: EXPLOATEAZĂ (folosește cea mai bună mișcare cunoscută)

        Acest echilibru este ESENȚIAL în ML!
        """
        state = self.state_to_string(board)
        valid_actions = self.get_valid_actions(board)

        if not valid_actions:
            return None

        # 🎲 EXPLORARE vs EXPLOATARE
        if training and random.random() < self.epsilon:
            # EXPLORARE: Încearcă ceva nou!
            action = random.choice(valid_actions)
            self.decision_reason = f"🎲 EXPLORARE: Încerc poziția {action + 1} la întâmplare (ε={self.epsilon:.2f})"
        else:
            # EXPLOATARE: Folosește cea mai bună mișcare cunoscută
            q_values = {action: self.q_table[(state, action)] for action in valid_actions}
            max_q = max(q_values.values())

            # Dacă sunt mai multe acțiuni cu aceeași valoare Q maximă, alege random dintre ele
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
            action = random.choice(best_actions)

            self.decision_reason = f"🧠 EXPLOATARE: Aleg poziția {action + 1} (Q-value: {max_q:.3f})"

        self.last_state = state
        self.last_action = action
        return action

    def update_q_value(self, reward, next_board=None):
        """
        🎓 ÎNVĂȚAREA PROPRIU-ZISĂ: Actualizează Q-Table-ul

        Formula Q-Learning:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

        Traducere în română:
        "Valoarea noua = Valoarea veche + rata_invatare * [recompensa + discount * cea_mai_buna_mutare_viitoare - valoarea_veche]"
        """
        if self.last_state is None or self.last_action is None:
            return

        current_q = self.q_table[(self.last_state, self.last_action)]

        # Calculează cea mai bună valoare Q pentru starea următoare
        if next_board is not None:
            next_state = self.state_to_string(next_board)
            next_valid_actions = self.get_valid_actions(next_board)
            if next_valid_actions:
                max_next_q = max(self.q_table[(next_state, action)] for action in next_valid_actions)
            else:
                max_next_q = 0
        else:
            max_next_q = 0  # Jocul s-a terminat

        # 🧮 FORMULA MAGICĂ Q-LEARNING
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(self.last_state, self.last_action)] = new_q

    def game_ended(self, reward):
        """
        🏁 Jocul s-a terminat - învață din rezultat
        """
        self.update_q_value(reward)
        self.total_games += 1

        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1

        # Reset pentru următorul joc
        self.last_state = None
        self.last_action = None

    def get_statistics(self):
        """
        📊 Statistici pentru monitorizarea progresului
        """
        if self.total_games == 0:
            return {"total": 0, "win_rate": 0, "loss_rate": 0, "draw_rate": 0}

        return {
            "total": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / self.total_games,
            "loss_rate": self.losses / self.total_games,
            "draw_rate": self.draws / self.total_games
        }

    def save_model(self, filename):
        """
        💾 Salvează "creierul" antrenat
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'stats': self.get_statistics(),
                'params': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.epsilon
                }
            }, f)

    def load_model(self, filename):
        """
        📁 Încarcă un "creier" antrenat anterior
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                stats = data['stats']
                self.total_games = stats['total']
                self.wins = stats['wins']
                self.losses = stats['losses']
                self.draws = stats['draws']
                return True
        except FileNotFoundError:
            return False


class TrainingThread(QThread):
    """
    🏃‍♂️ Thread separată pentru antrenament - nu blochează interfața
    """
    progress_updated = pyqtSignal(int, dict)  # progres, statistici
    training_completed = pyqtSignal()

    def __init__(self, agent, episodes=1000):
        super().__init__()
        self.agent = agent
        self.episodes = episodes
        self.should_stop = False

    def run(self):
        """
        🎯 Antrenamentul propriu-zis - self-play
        """
        for episode in range(self.episodes):
            if self.should_stop:
                break

            # Joacă un joc complet împotriva unui adversar random
            self.play_training_game()

            # Actualizează progresul la fiecare 100 de jocuri
            if episode % 100 == 0:
                stats = self.agent.get_statistics()
                self.progress_updated.emit(episode, stats)

        self.training_completed.emit()

    def play_training_game(self):
        """
        🎮 Joacă un joc de antrenament împotriva unui adversar random
        """
        board = [''] * 9
        ai_turn = random.choice([True, False])  # Cine începe aleator

        while True:
            # Verifică dacă jocul s-a terminat
            winner = self.check_winner(board)
            if winner is not None:
                # Recompensă finală
                if winner == 'O':  # AI-ul a câștigat
                    self.agent.game_ended(1.0)
                elif winner == 'X':  # AI-ul a pierdut
                    self.agent.game_ended(-1.0)
                else:  # Egalitate
                    self.agent.game_ended(0.0)
                break

            # Egalitate (tabla plină)
            if '' not in board:
                self.agent.game_ended(0.0)
                break

            if ai_turn:
                # Rândul AI-ului
                action = self.agent.choose_action(board, training=True)
                if action is not None and board[action] == '':
                    board[action] = 'O'
                    self.agent.update_q_value(-0.01)  # Mică penalizare pentru că jocul continuă
                else:
                    # Mutare invalidă - penalizare mare
                    self.agent.game_ended(-0.5)
                    break
            else:
                # Rândul adversarului random
                valid_moves = [i for i, cell in enumerate(board) if cell == '']
                if valid_moves:
                    move = random.choice(valid_moves)
                    board[move] = 'X'

            ai_turn = not ai_turn

    def check_winner(self, board):
        """
        🏆 Verifică câștigătorul
        """
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rânduri
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # coloane
            [0, 4, 8], [2, 4, 6]              # diagonale
        ]

        for condition in win_conditions:
            if board[condition[0]] == board[condition[1]] == board[condition[2]] != '':
                return board[condition[0]]
        return None

    def stop_training(self):
        self.should_stop = True


class MLExplanationDialog(QDialog):
    """
    🎓 Dialog educațional pentru explicarea deciziilor ML vs reguli simple
    """

    def __init__(self, ml_agent, board_state, chosen_move, parent=None):
        super().__init__(parent)
        self.ml_agent = ml_agent
        self.board_state = board_state[:]
        self.chosen_move = chosen_move
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🧠 AI cu Machine Learning vs AI cu Reguli Simple')
        self.setGeometry(250, 150, 800, 700)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #f8f9fa, stop: 1 #e9ecef);
            }
            QLabel {
                color: #2c3e50;
                font-family: Arial;
            }
        """)

        layout = QVBoxLayout()

        # Titlu
        title = QLabel('🤖 Comparație: AI cu Reguli vs AI cu Machine Learning')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            background-color: #2c3e50;
            padding: 15px;
            border-radius: 8px;
            margin: 10px;
        """)
        layout.addWidget(title)

        # Tab widget pentru comparație
        tabs = QTabWidget()

        # Tab 1: AI cu reguli simple
        rules_tab = self.create_rules_explanation()
        tabs.addTab(rules_tab, "🔧 AI cu Reguli Simple")

        # Tab 2: AI cu ML
        ml_tab = self.create_ml_explanation()
        tabs.addTab(ml_tab, "🧠 AI cu Machine Learning")

        # Tab 3: Comparația
        comparison_tab = self.create_comparison()
        tabs.addTab(comparison_tab, "⚖️ Comparație")

        layout.addWidget(tabs)

        # Buton închidere
        close_btn = QPushButton('Înțeleg diferența! 🎓')
        close_btn.setStyleSheet("""
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def create_rules_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Explicație reguli simple
        explanation = QLabel("""
🔧 AI-ul cu REGULI SIMPLE (din xo_game.py):

📋 ALGORITM FIX - 3 pași:
1. Verific dacă pot câștiga → câștig
2. Verific dacă adversarul poate câștiga → blochez
3. Altfel → aleg random

✅ AVANTAJE:
• Simplu de înțeles și implementat
• Predictibil și controlabil
• Nu are nevoie de antrenament
• Funcționează imediat

❌ DEZAVANTAJE:
• Nu învață nimic din experiență
• Nu dezvoltă strategii avansate
• Nu se adaptează la stilul adversarului
• Limitat la aceste 3 reguli pentru totdeauna

🤖 CONCLUZIE:
Acest AI este ca un robot care execută aceleași instrucțiuni.
Nu poate deveni mai bun decât programatorul său.
        """)

        explanation.setStyleSheet("""
            font-size: 13px;
            background-color: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #ff9800;
        """)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        widget.setLayout(layout)
        return widget

    def create_ml_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Statistici AI
        stats = self.ml_agent.get_statistics()

        explanation = QLabel(f"""
🧠 AI-ul cu MACHINE LEARNING (Q-Learning):

📊 STATISTICI ACTUALE:
• Jocuri jucate: {stats['total']}
• Rata de câștig: {stats['win_rate']:.1%}
• Q-Table entries: {len(self.ml_agent.q_table)}

🎯 ALGORITM ADAPTIV:
• Încearcă diferite mutări (explorare)
• Învață din rezultate (recompense/penalizări)
• Își construiește o "memorie" (Q-Table)
• Devine mai bun cu fiecare joc

✅ AVANTAJE:
• Învață și se îmbunătățește constant
• Poate descoperi strategii necunoscute
• Se adaptează la stilul adversarului
• Poate deveni mai bun decât programatorul

❌ DEZAVANTAJE:
• Are nevoie de mult antrenament
• Deciziile nu sunt transparente
• Poate face greșeli în timpul învățării
• Necesită resurse computaționale

🤖 CONCLUZIE:
Acest AI este ca un elev care învață din experiență.
Poate deveni mai bun decât ne-am imaginat!

🔍 ULTIMA DECIZIE: {self.ml_agent.decision_reason}
        """)

        explanation.setStyleSheet("""
            font-size: 13px;
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #4caf50;
        """)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        widget.setLayout(layout)
        return widget

    def create_comparison(self):
        widget = QWidget()
        layout = QVBoxLayout()

        comparison = QLabel("""
⚖️ COMPARAȚIA FUNDAMENTALĂ:

🏗️ CONSTRUIREA:
• Reguli Simple: Programatorul scrie reguli explicite
• Machine Learning: AI-ul își descoperă singur regulile

🧠 "GÂNDIREA":
• Reguli Simple: "Am această situație → fac această acțiune"
• Machine Learning: "În situații similare, această acțiune mi-a adus succes"

📈 EVOLUȚIA:
• Reguli Simple: Rămâne la fel pentru totdeauna
• Machine Learning: Devine mai bun cu timpul

🔍 TRANSPARENȚA:
• Reguli Simple: Știm exact de ce face fiecare mutare
• Machine Learning: E misterios, chiar și pentru creatori

🌍 APLICAREA ÎN REALITATE:

🔧 Unde folosim AI cu REGULI:
• Sisteme de siguranță (avioane, mașini)
• Protocoale medicale
• Sisteme de control industrial

🧠 Unde folosim MACHINE LEARNING:
• Recunoașterea vocii (Siri, Alexa)
• Recomandări (Netflix, YouTube)
• Traduceri automate
• Mașini autonome
• Diagnostic medical avansat

💭 PARADOXUL AI-ULUI MODERN:
Cu cât AI-ul devine mai puternic, cu atât devine mai misterios!
ChatGPT și alte AI-uri moderne sunt atât de complexe încât
nici creatorii lor nu înțeleg complet cum funcționează.
        """)

        comparison.setStyleSheet("""
            font-size: 12px;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #9b59b6;
        """)
        comparison.setWordWrap(True)
        layout.addWidget(comparison)

        widget.setLayout(layout)
        return widget


class TicTacToeAI(QWidget):
    """
    🎮 Jocul principal cu AI Machine Learning
    """

    def __init__(self):
        super().__init__()
        self.ml_agent = QLearningAgent()
        self.training_thread = None

        # Încarcă modelul antrenat dacă există
        if self.ml_agent.load_model('xo_ai_model.pkl'):
            print("✅ Model antrenat încărcat cu succes!")
        else:
            print("ℹ️ Nu s-a găsit model antrenat. Începe cu Q-Table gol.")

        self.initUI()

    def initUI(self):
        self.setWindowTitle('🧠 X și O - AI cu Machine Learning (Q-Learning)')
        self.setGeometry(50, 50, 900, 700)

        # Layout principal cu splitter
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # Partea stângă - jocul
        game_widget = self.create_game_panel()
        splitter.addWidget(game_widget)

        # Partea dreaptă - antrenament și statistici
        training_widget = self.create_training_panel()
        splitter.addWidget(training_widget)

        splitter.setSizes([500, 400])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Inițializează jocul
        self.reset_game()

    def create_game_panel(self):
        """
        🎮 Panelul cu jocul propriu-zis
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # Titlu
        title = QLabel('🎯 Joacă împotriva AI-ului cu Machine Learning!')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel('Rândul tău (X)')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            background-color: #3498db;
            color: white;
            padding: 10px;
            border-radius: 5px;
        """)
        layout.addWidget(self.status_label)

        # AI thinking
        self.thinking_label = QLabel('')
        self.thinking_label.setAlignment(Qt.AlignCenter)
        self.thinking_label.setStyleSheet("""
            font-size: 14px;
            background-color: #e67e22;
            color: white;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
        """)
        self.thinking_label.hide()
        layout.addWidget(self.thinking_label)

        # Grid de joc
        grid_layout = QVBoxLayout()
        self.buttons = []
        for i in range(3):
            row_layout = QHBoxLayout()
            for j in range(3):
                button = QPushButton('')
                button.setFixedSize(80, 80)
                button.setStyleSheet("""
                    QPushButton {
                        font-size: 24px;
                        font-weight: bold;
                        border: 2px solid #34495e;
                        border-radius: 8px;
                        background-color: #ecf0f1;
                    }
                    QPushButton:hover {
                        background-color: #bdc3c7;
                    }
                """)
                index = i * 3 + j
                button.clicked.connect(lambda _, idx=index: self.on_button_click(idx))
                self.buttons.append(button)
                row_layout.addWidget(button)
            grid_layout.addLayout(row_layout)
        layout.addLayout(grid_layout)

        # Butoane control
        control_layout = QHBoxLayout()

        self.new_game_btn = QPushButton('🔄 Joc Nou')
        self.new_game_btn.clicked.connect(self.reset_game)

        self.explain_btn = QPushButton('🧠 Explică decizia AI')
        self.explain_btn.clicked.connect(self.explain_ai_decision)
        self.explain_btn.setEnabled(False)

        control_layout.addWidget(self.new_game_btn)
        control_layout.addWidget(self.explain_btn)
        layout.addLayout(control_layout)

        widget.setLayout(layout)
        return widget

    def create_training_panel(self):
        """
        🏃‍♂️ Panelul pentru antrenament și monitorizare
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # Titlu antrenament
        training_title = QLabel('🎓 Antrenament Machine Learning')
        training_title.setAlignment(Qt.AlignCenter)
        training_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(training_title)

        # Statistici actuale
        self.stats_label = QLabel(self.get_stats_text())
        self.stats_label.setStyleSheet("""
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        """)
        layout.addWidget(self.stats_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Butoane antrenament
        training_layout = QHBoxLayout()

        self.train_btn = QPushButton('🏃‍♂️ Antrenează (1000 jocuri)')
        self.train_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton('⏹️ Stop')
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)

        training_layout.addWidget(self.train_btn)
        training_layout.addWidget(self.stop_btn)
        layout.addLayout(training_layout)

        # Salvare/Încărcare model
        model_layout = QHBoxLayout()

        save_btn = QPushButton('💾 Salvează AI antrenat')
        save_btn.clicked.connect(self.save_model)

        load_btn = QPushButton('📁 Încarcă AI antrenat')
        load_btn.clicked.connect(self.load_model)

        model_layout.addWidget(save_btn)
        model_layout.addWidget(load_btn)
        layout.addLayout(model_layout)

        # Explicații educaționale
        education_text = QLabel("""
🎯 CE SE ÎNTÂMPLĂ ÎN ANTRENAMENT:

1. AI-ul joacă mii de jocuri împotriva unui adversar random
2. Pentru fiecare mutare, primește recompense:
   • +1 pentru câștig
   • -1 pentru pierdere
   • 0 pentru egalitate
   • -0.01 pentru continuarea jocului
3. Construiește o "memorie" (Q-Table) cu ce să facă în fiecare situație
4. Balansează explorarea (mutări noi) cu exploatarea (mutări cunoscute ca bune)

💡 Cu cât joacă mai mult, cu atât devine mai inteligent!
        """)
        education_text.setStyleSheet("""
            background-color: #e8f5e8;
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            border-left: 4px solid #4caf50;
        """)
        education_text.setWordWrap(True)
        layout.addWidget(education_text)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def get_stats_text(self):
        """
        📊 Text cu statisticile curente
        """
        stats = self.ml_agent.get_statistics()
        return f"""
📊 STATISTICI AI:
• Jocuri jucate: {stats['total']}
• Victorii: {stats['wins']} ({stats['win_rate']:.1%})
• Înfrângeri: {stats['losses']} ({stats['loss_rate']:.1%})
• Egalități: {stats['draws']} ({stats['draw_rate']:.1%})
• Intrări în Q-Table: {len(self.ml_agent.q_table)}
        """

    def reset_game(self):
        """
        🔄 Resetează jocul
        """
        self.board = [''] * 9
        self.player_turn = True
        self.game_over = False
        self.last_ai_move = None

        for button in self.buttons:
            button.setText('')
            button.setEnabled(True)
            button.setStyleSheet("""
                QPushButton {
                    font-size: 24px;
                    font-weight: bold;
                    border: 2px solid #34495e;
                    border-radius: 8px;
                    background-color: #ecf0f1;
                }
                QPushButton:hover {
                    background-color: #bdc3c7;
                }
            """)

        self.status_label.setText('Rândul tău (X)')
        self.thinking_label.hide()
        self.explain_btn.setEnabled(False)

    def on_button_click(self, index):
        """
        🖱️ Click pe butonul din grid
        """
        if not self.player_turn or self.game_over or self.board[index] != '':
            return

        # Mutarea jucătorului
        self.board[index] = 'X'
        self.buttons[index].setText('X')
        self.buttons[index].setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 8px;
                background-color: #3498db;
                color: white;
            }
        """)

        # Verifică sfârșitul jocului
        if self.check_game_end():
            return

        # Rândul AI-ului
        self.player_turn = False
        self.status_label.setText('🤖 AI-ul se gândește...')
        self.thinking_label.setText('🧠 Analizez Q-Table-ul...')
        self.thinking_label.show()

        # Simulează "gândirea" AI-ului
        QTimer.singleShot(1500, self.ai_move)

    def ai_move(self):
        """
        🤖 Mutarea AI-ului
        """
        action = self.ml_agent.choose_action(self.board, training=False)

        if action is not None:
            self.board[action] = 'O'
            self.buttons[action].setText('O')
            self.buttons[action].setStyleSheet("""
                QPushButton {
                    font-size: 24px;
                    font-weight: bold;
                    border: 2px solid #34495e;
                    border-radius: 8px;
                    background-color: #e74c3c;
                    color: white;
                }
            """)

            self.last_ai_move = action
            self.explain_btn.setEnabled(True)

            # Afișează decizia AI-ului
            self.thinking_label.setText(self.ml_agent.decision_reason)
            QTimer.singleShot(3000, self.thinking_label.hide)

        if not self.check_game_end():
            self.player_turn = True
            self.status_label.setText('Rândul tău (X)')

    def check_game_end(self):
        """
        🏁 Verifică sfârșitul jocului
        """
        winner = self.check_winner()

        if winner:
            self.game_over = True
            self.thinking_label.hide()

            if winner == 'X':
                self.status_label.setText('🎉 Ai câștigat!')
                self.ml_agent.game_ended(-1.0)  # AI-ul a pierdut
            elif winner == 'O':
                self.status_label.setText('🤖 AI-ul a câștigat!')
                self.ml_agent.game_ended(1.0)   # AI-ul a câștigat

            for button in self.buttons:
                button.setEnabled(False)

            # Actualizează statisticile
            self.stats_label.setText(self.get_stats_text())
            return True

        elif '' not in self.board:
            # Egalitate
            self.game_over = True
            self.status_label.setText('🤝 Egalitate!')
            self.ml_agent.game_ended(0.0)
            self.thinking_label.hide()

            for button in self.buttons:
                button.setEnabled(False)

            self.stats_label.setText(self.get_stats_text())
            return True

        return False

    def check_winner(self):
        """
        🏆 Verifică câștigătorul
        """
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rânduri
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # coloane
            [0, 4, 8], [2, 4, 6]              # diagonale
        ]

        for condition in win_conditions:
            if (self.board[condition[0]] == self.board[condition[1]] ==
                self.board[condition[2]] != ''):
                return self.board[condition[0]]
        return None

    def explain_ai_decision(self):
        """
        🧠 Explică decizia AI-ului
        """
        if self.last_ai_move is not None:
            dialog = MLExplanationDialog(self.ml_agent, self.board, self.last_ai_move, self)
            dialog.exec_()

    def start_training(self):
        """
        🏃‍♂️ Începe antrenamentul
        """
        self.training_thread = TrainingThread(self.ml_agent, episodes=1000)
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_completed.connect(self.training_finished)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(1000)

        self.training_thread.start()

    def update_training_progress(self, episode, stats):
        """
        📊 Actualizează progresul antrenamentului
        """
        self.progress_bar.setValue(episode)
        self.stats_label.setText(self.get_stats_text())

    def training_finished(self):
        """
        🏁 Antrenamentul s-a terminat
        """
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.stats_label.setText(self.get_stats_text())

        # Salvează automat modelul
        self.save_model()

    def stop_training(self):
        """
        ⏹️ Oprește antrenamentul
        """
        if self.training_thread:
            self.training_thread.stop_training()

    def save_model(self):
        """
        💾 Salvează modelul antrenat
        """
        self.ml_agent.save_model('xo_ai_model.pkl')
        self.status_label.setText('💾 Model salvat cu succes!')
        QTimer.singleShot(2000, lambda: self.status_label.setText('Rândul tău (X)' if self.player_turn else '🤖 AI-ul se gândește...'))

    def load_model(self):
        """
        📁 Încarcă un model antrenat
        """
        if self.ml_agent.load_model('xo_ai_model.pkl'):
            self.stats_label.setText(self.get_stats_text())
            self.status_label.setText('📁 Model încărcat cu succes!')
            QTimer.singleShot(2000, lambda: self.status_label.setText('Rândul tău (X)' if self.player_turn else '🤖 AI-ul se gândește...'))
        else:
            self.status_label.setText('⚠️ Nu s-a găsit model salvat!')
            QTimer.singleShot(2000, lambda: self.status_label.setText('Rândul tău (X)' if self.player_turn else '🤖 AI-ul se gândește...'))


if __name__ == '__main__':
    print("""
    🧠 X și O cu Machine Learning - Pornire aplicație
    ================================================

    Această aplicație demonstrează diferența dintre:
    1. AI cu reguli simple (din xo_game.py)
    2. AI cu Machine Learning (Q-Learning)

    💡 Sugestii:
    1. Joacă câteva jocuri pentru a vedea cum se comportă AI-ul inițial
    2. Antrenează AI-ul (butonul "Antrenează")
    3. Joacă din nou și observă îmbunătățirea
    4. Folosește butonul "Explică decizia" pentru a înțelege diferențele

    🎯 Scopul educațional: Să înțelegi diferența dintre programarea
    tradițională (reguli explicite) și Machine Learning (învățare din experiență)
    """)

    app = QApplication(sys.argv)
    game = TicTacToeAI()
    game.show()
    sys.exit(app.exec_())