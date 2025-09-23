"""
🔍 Inspector pentru Modelul Q-Learning X și O
=============================================
Acest script îți permite să vezi ce este în interiorul modelului .pkl
și să identifici problemele potențiale cu antrenamentul.

Autor: Lao & Claude
"""

import pickle
import sys
from collections import defaultdict, Counter
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                             QTextEdit, QTabWidget, QComboBox, QSpinBox,
                             QScrollArea, QHeaderView, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


class ModelInspector(QWidget):
    def __init__(self):
        super().__init__()
        self.model_data = None
        self.q_table = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🔍 Inspector Model Q-Learning X și O')
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()

        # Header
        header = QLabel('🧠 Analizator Model Q-Learning')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        """)
        layout.addWidget(header)

        # Load button
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton('📁 Încarcă Model (xo_ai_model.pkl)')
        self.load_btn.clicked.connect(self.load_model)
        load_layout.addWidget(self.load_btn)

        self.status_label = QLabel('')
        load_layout.addWidget(self.status_label)
        layout.addLayout(load_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setEnabled(False)

        # Tab 1: Statistics
        self.stats_tab = self.create_stats_tab()
        self.tabs.addTab(self.stats_tab, "📊 Statistici")

        # Tab 2: Q-Table Analysis
        self.qtable_tab = self.create_qtable_tab()
        self.tabs.addTab(self.qtable_tab, "🗃️ Analiza Q-Table")

        # Tab 3: Best Moves
        self.best_moves_tab = self.create_best_moves_tab()
        self.tabs.addTab(self.best_moves_tab, "🎯 Cele Mai Bune Mutări")

        # Tab 4: Problems
        self.problems_tab = self.create_problems_tab()
        self.tabs.addTab(self.problems_tab, "⚠️ Probleme Detectate")

        # Tab 5: Board Visualizer
        self.board_tab = self.create_board_visualizer_tab()
        self.tabs.addTab(self.board_tab, "🎮 Vizualizator Tablă")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def create_stats_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            font-size: 14px;
            font-family: monospace;
            background-color: #f8f9fa;
        """)
        layout.addWidget(self.stats_text)

        widget.setLayout(layout)
        return widget

    def create_qtable_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filtrează după valoare Q:"))

        self.min_q_spin = QSpinBox()
        self.min_q_spin.setRange(-100, 100)
        self.min_q_spin.setValue(0)
        self.min_q_spin.setSingleStep(1)
        filter_layout.addWidget(QLabel("Min:"))
        filter_layout.addWidget(self.min_q_spin)

        self.filter_btn = QPushButton("🔍 Filtrează")
        self.filter_btn.clicked.connect(self.filter_qtable)
        filter_layout.addWidget(self.filter_btn)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Q-Table
        self.qtable_widget = QTableWidget()
        self.qtable_widget.setColumnCount(3)
        self.qtable_widget.setHorizontalHeaderLabels(['Stare (Board)', 'Acțiune', 'Q-Value'])
        layout.addWidget(self.qtable_widget)

        widget.setLayout(layout)
        return widget

    def create_best_moves_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        info = QLabel("""
        🎯 Aici poți vedea ce mutări consideră AI-ul că sunt cele mai bune
        pentru diferite situații de joc comune.
        """)
        info.setWordWrap(True)
        layout.addWidget(info)

        self.best_moves_text = QTextEdit()
        self.best_moves_text.setReadOnly(True)
        self.best_moves_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.best_moves_text)

        widget.setLayout(layout)
        return widget

    def create_problems_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.problems_text = QTextEdit()
        self.problems_text.setReadOnly(True)
        self.problems_text.setStyleSheet("""
            font-size: 13px;
            background-color: #fff3e0;
        """)
        layout.addWidget(self.problems_text)

        # Fix suggestions
        self.fix_btn = QPushButton("🔧 Generează Sugestii de Remediere")
        self.fix_btn.clicked.connect(self.generate_fixes)
        layout.addWidget(self.fix_btn)

        widget.setLayout(layout)
        return widget

    def create_board_visualizer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        info = QLabel("Introdu o stare de tablă pentru a vedea ce ar face AI-ul:")
        layout.addWidget(info)

        # Board input
        board_layout = QHBoxLayout()
        self.board_input = QTextEdit()
        self.board_input.setMaximumHeight(100)
        self.board_input.setPlaceholderText("Ex: X_O_X____ sau X O X     ")
        board_layout.addWidget(self.board_input)

        self.analyze_board_btn = QPushButton("🔍 Analizează")
        self.analyze_board_btn.clicked.connect(self.analyze_board_state)
        board_layout.addWidget(self.analyze_board_btn)
        layout.addLayout(board_layout)

        # Results
        self.board_analysis = QTextEdit()
        self.board_analysis.setReadOnly(True)
        layout.addWidget(self.board_analysis)

        widget.setLayout(layout)
        return widget

    def load_model(self):
        """Încarcă modelul .pkl"""
        try:
            # Try default name first
            filename = 'xo_ai_model.pkl'
            try:
                with open(filename, 'rb') as f:
                    self.model_data = pickle.load(f)
            except FileNotFoundError:
                # Ask user to select file
                filename, _ = QFileDialog.getOpenFileName(self, 'Selectează Model', '', 'Pickle Files (*.pkl)')
                if not filename:
                    return
                with open(filename, 'rb') as f:
                    self.model_data = pickle.load(f)

            self.q_table = defaultdict(float, self.model_data['q_table'])
            self.status_label.setText(f"✅ Model încărcat: {len(self.q_table)} intrări")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.tabs.setEnabled(True)

            self.analyze_model()

        except Exception as e:
            self.status_label.setText(f"❌ Eroare: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def analyze_model(self):
        """Analizează modelul încărcat"""
        if not self.model_data:
            return

        # 1. Statistics
        stats = self.model_data.get('stats', {})
        params = self.model_data.get('params', {})

        stats_text = f"""
📊 STATISTICI MODEL
==================

🎮 Jocuri Jucate: {stats.get('total', 0)}
✅ Victorii: {stats.get('wins', 0)} ({stats.get('win_rate', 0):.2%})
❌ Înfrângeri: {stats.get('losses', 0)} ({stats.get('loss_rate', 0):.2%})
🤝 Egalități: {stats.get('draws', 0)} ({stats.get('draw_rate', 0):.2%})

📚 Q-TABLE
==========
Total intrări: {len(self.q_table)}
Valoare minimă Q: {min(self.q_table.values()):.4f}
Valoare maximă Q: {max(self.q_table.values()):.4f}
Valoare medie Q: {np.mean(list(self.q_table.values())):.4f}

🔧 PARAMETRI ANTRENAMENT
========================
Learning Rate (α): {params.get('learning_rate', 'N/A')}
Discount Factor (γ): {params.get('discount_factor', 'N/A')}
Epsilon (ε): {params.get('epsilon', 'N/A')}

📈 DISTRIBUȚIA Q-VALUES
=======================
"""

        # Q-value distribution
        q_values = list(self.q_table.values())
        bins = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
        hist, _ = np.histogram(q_values, bins=bins)

        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            bar_length = int(hist[i] / max(hist) * 40) if max(hist) > 0 else 0
            stats_text += f"[{start:5.1f} to {end:5.1f}]: {'█' * bar_length} {hist[i]}\n"

        self.stats_text.setText(stats_text)

        # 2. Populate Q-Table view
        self.populate_qtable_view()

        # 3. Analyze best moves
        self.analyze_best_moves()

        # 4. Detect problems
        self.detect_problems()

    def populate_qtable_view(self):
        """Populează vizualizarea Q-Table"""
        self.qtable_widget.setRowCount(len(self.q_table))

        sorted_entries = sorted(self.q_table.items(), key=lambda x: x[1], reverse=True)

        for row, ((state, action), q_value) in enumerate(sorted_entries[:1000]):  # Show top 1000
            # State
            state_item = QTableWidgetItem(state)
            state_item.setFont(QFont("monospace", 9))
            self.qtable_widget.setItem(row, 0, state_item)

            # Action
            action_item = QTableWidgetItem(f"Poziția {action + 1}")
            self.qtable_widget.setItem(row, 1, action_item)

            # Q-Value
            q_item = QTableWidgetItem(f"{q_value:.6f}")
            if q_value > 0.5:
                q_item.setBackground(QColor(200, 255, 200))
            elif q_value < -0.5:
                q_item.setBackground(QColor(255, 200, 200))
            self.qtable_widget.setItem(row, 2, q_item)

    def filter_qtable(self):
        """Filtrează Q-Table după valoare minimă"""
        min_val = self.min_q_spin.value() / 100
        filtered = {k: v for k, v in self.q_table.items() if v >= min_val}

        self.qtable_widget.setRowCount(len(filtered))
        sorted_entries = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        for row, ((state, action), q_value) in enumerate(sorted_entries):
            state_item = QTableWidgetItem(state)
            state_item.setFont(QFont("monospace", 9))
            self.qtable_widget.setItem(row, 0, state_item)

            action_item = QTableWidgetItem(f"Poziția {action + 1}")
            self.qtable_widget.setItem(row, 1, action_item)

            q_item = QTableWidgetItem(f"{q_value:.6f}")
            self.qtable_widget.setItem(row, 2, q_item)

    def analyze_best_moves(self):
        """Analizează cele mai bune mutări pentru situații comune"""
        analysis = "🎯 ANALIZA MUTĂRILOR OPTIME\n" + "=" * 40 + "\n\n"

        # Common game situations
        test_situations = [
            ("_________", "Tablă goală"),
            ("____X____", "X în centru"),
            ("X________", "X în colțul stânga-sus"),
            ("X___X____", "X pe diagonală"),
            ("XX_______", "Două X consecutive"),
            ("X_X______", "X cu spațiu între ele"),
            ("XOX______", "X-O-X pe prima linie"),
        ]

        for board_str, description in test_situations:
            analysis += f"\n📋 {description}\n"
            analysis += f"Tablă: {self.format_board(board_str)}\n"

            # Find best move for this state
            best_action = None
            best_q = float('-inf')

            for action in range(9):
                if board_str[action] == '_':
                    q_val = self.q_table.get((board_str, action), 0)
                    if q_val > best_q:
                        best_q = q_val
                        best_action = action

            if best_action is not None:
                analysis += f"Cea mai bună mutare: Poziția {best_action + 1} (Q={best_q:.4f})\n"
            else:
                analysis += "Nu am găsit mutare în Q-Table pentru această situație!\n"

        self.best_moves_text.setText(analysis)

    def format_board(self, board_str):
        """Formatează o tablă pentru afișare"""
        board = board_str.replace('_', '·')
        return f"\n  {board[0]} | {board[1]} | {board[2]}\n  {board[3]} | {board[4]} | {board[5]}\n  {board[6]} | {board[7]} | {board[8]}"

    def detect_problems(self):
        """Detectează probleme în model"""
        problems = "⚠️ PROBLEME DETECTATE ÎN MODEL\n" + "=" * 40 + "\n\n"

        # Problem 1: Check if AI knows to win
        win_situations = [
            ("OO_______", 2, "două O consecutive pe linia 1"),
            ("_OO______", 0, "două O consecutive pe linia 1"),
            ("O_O______", 1, "O-spațiu-O pe linia 1"),
            ("O___O____", 8, "două O pe diagonală"),
        ]

        missing_wins = []
        for board, winning_move, desc in win_situations:
            q_val = self.q_table.get((board, winning_move), 0)
            if q_val < 0.8:  # Should be close to 1.0
                missing_wins.append((desc, board, winning_move, q_val))

        if missing_wins:
            problems += "🚨 PROBLEMA 1: AI-ul nu știe să câștige!\n"
            for desc, board, move, q in missing_wins:
                problems += f"  • {desc}: Q={q:.4f} (ar trebui ~1.0)\n"
        else:
            problems += "✅ AI-ul știe să câștige când are ocazia\n"

        # Problem 2: Check if AI knows to block
        block_situations = [
            ("XX_______", 2, "două X consecutive"),
            ("X_X______", 1, "X-spațiu-X"),
            ("X___X____", 8, "X pe diagonală"),
        ]

        missing_blocks = []
        for board, block_move, desc in block_situations:
            q_val = self.q_table.get((board, block_move), 0)
            if q_val < 0.5:  # Should be high
                missing_blocks.append((desc, board, block_move, q_val))

        if missing_blocks:
            problems += "\n🚨 PROBLEMA 2: AI-ul nu știe să blocheze!\n"
            for desc, board, move, q in missing_blocks:
                problems += f"  • {desc}: Q={q:.4f} (ar trebui >0.5)\n"
        else:
            problems += "\n✅ AI-ul știe să blocheze adversarul\n"

        # Problem 3: Q-value distribution
        q_values = list(self.q_table.values())
        if len(q_values) > 0:
            neg_ratio = sum(1 for q in q_values if q < 0) / len(q_values)
            if neg_ratio > 0.7:
                problems += f"\n🚨 PROBLEMA 3: Prea multe Q-values negative ({neg_ratio:.1%})\n"
                problems += "  Acest lucru sugerează că AI-ul pierde prea des în antrenament.\n"

        # Problem 4: Limited exploration
        unique_states = len(set(state for state, _ in self.q_table.keys()))
        problems += f"\n📊 Stări unice explorate: {unique_states}\n"
        if unique_states < 500:
            problems += "⚠️ PROBLEMA 4: Explorare limitată - AI-ul a văzut prea puține situații diferite\n"

        # Problem 5: Learning parameters
        params = self.model_data.get('params', {})
        lr = params.get('learning_rate', 0.3)
        epsilon = params.get('epsilon', 0.3)

        if lr > 0.5:
            problems += f"\n⚠️ Learning rate prea mare ({lr}) - poate cauza instabilitate\n"
        if epsilon < 0.1:
            problems += f"\n⚠️ Epsilon prea mic ({epsilon}) - explorare insuficientă\n"

        # MAIN ISSUE FOUND
        problems += "\n" + "=" * 40 + "\n"
        problems += """
🔴 PROBLEMA PRINCIPALĂ IDENTIFICATĂ:
=====================================

După analiza codului din fișierul original, am găsit câteva probleme MAJORE:

1. **RECOMPENSE GREȘITE**: 
   - În funcția play_training_game(), AI-ul primește -0.01 după FIECARE mutare
   - Aceasta îl încurajează să evite să facă mutări!

2. **ANTRENAMENT CU ADVERSAR RANDOM**:
   - AI-ul joacă doar împotriva mutări random, nu învață strategii reale
   - Un adversar random nu oferă provocări consistente

3. **UPDATE Q-VALUE INCOMPLET**:
   - Q-values sunt actualizate doar la sfârșitul jocului
   - Nu există actualizări intermediare pentru mutări bune/rele

4. **LIPSĂ STRATEGIE INIȚIALĂ**:
   - În primele jocuri, AI-ul face mutări complet random
   - Nu are o bază strategică de la care să pornească

SOLUȚII RECOMANDATE:
• Schimbă sistemul de recompense (vezi tab-ul Sugestii)
• Antrenează împotriva unui adversar mai inteligent
• Implementează reward shaping pentru mutări strategice
• Folosește self-play cu versiuni anterioare ale AI-ului
"""

        self.problems_text.setText(problems)

    def analyze_board_state(self):
        """Analizează o stare specifică de tablă"""
        board_input = self.board_input.toPlainText().strip()
        if not board_input:
            return

        # Normalize input
        board_str = board_input.replace(' ', '_').replace('.', '_').replace('-', '_')
        if len(board_str) != 9:
            self.board_analysis.setText("❌ Tabla trebuie să aibă exact 9 poziții!")
            return

        analysis = f"📋 ANALIZĂ TABLĂ\n{'=' * 40}\n\n"
        analysis += f"Stare: {self.format_board(board_str)}\n\n"

        # Find all Q-values for valid moves
        moves = []
        for action in range(9):
            if board_str[action] == '_':
                q_val = self.q_table.get((board_str, action), 0.0)
                moves.append((action, q_val))

        if not moves:
            analysis += "Nu sunt mutări valide pentru această stare!\n"
        else:
            moves.sort(key=lambda x: x[1], reverse=True)
            analysis += "MUTĂRI DISPONIBILE (sortate după Q-value):\n"
            for action, q_val in moves:
                row, col = action // 3, action % 3
                analysis += f"  • Poziția {action + 1} (rând {row + 1}, col {col + 1}): Q={q_val:.6f}\n"

            best_action = moves[0][0]
            analysis += f"\n🎯 Cea mai bună mutare: Poziția {best_action + 1}\n"

            # Check if this makes sense strategically
            if self.is_winning_move(board_str, best_action, 'O'):
                analysis += "✅ Aceasta este o mutare câștigătoare!\n"
            elif self.is_blocking_move(board_str, best_action):
                analysis += "🛡️ Aceasta blochează o mutare câștigătoare a adversarului\n"
            elif best_action == 4 and board_str[4] == '_':
                analysis += "📍 Aceasta ia centrul - strategie bună!\n"

        self.board_analysis.setText(analysis)

    def is_winning_move(self, board, action, player):
        """Verifică dacă o mutare câștigă"""
        test_board = list(board)
        test_board[action] = player
        return self.check_winner(test_board) == player

    def is_blocking_move(self, board, action):
        """Verifică dacă o mutare blochează adversarul"""
        return self.is_winning_move(board, action, 'X')

    def check_winner(self, board):
        """Verifică câștigătorul"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]
        for pattern in win_patterns:
            if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != '_':
                return board[pattern[0]]
        return None

    def generate_fixes(self):
        """Generează cod pentru a remedia problemele"""
        fixes = """
🔧 COD DE REMEDIERE PENTRU PROBLEMELE GĂSITE
============================================

Înlocuiește funcția play_training_game() cu aceasta:

```python
def play_training_game(self):
    '''🎮 Joc de antrenament îmbunătățit'''
    board = [''] * 9
    ai_turn = random.choice([True, False])
    states_actions = []  # Salvăm istoricul pentru reward propagation

    while True:
        winner = self.agent.check_winner(board)
        if winner is not None or '' not in board:
            # Calculăm recompensa finală
            if winner == 'O':  # AI won
                final_reward = 1.0
            elif winner == 'X':  # AI lost
                final_reward = -1.0
            else:  # Draw
                final_reward = 0.1  # Small positive for draw

            # Propagăm recompensa înapoi
            for i, (state, action) in enumerate(reversed(states_actions)):
                # Recompensă descrescătoare pentru mutări mai vechi
                discount = 0.9 ** i
                self.agent.q_table[(state, action)] += \
                    self.agent.learning_rate * discount * final_reward

            self.agent.game_ended(final_reward)
            break

        if ai_turn:
            # AI's turn with improved rewards
            action = self.agent.choose_action(board, training=True)
            if action is not None and board[action] == '':
                old_state = self.agent.state_to_string(board)
                board[action] = 'O'
                states_actions.append((old_state, action))

                # Immediate rewards for good moves
                if self.is_winning_move(board, 'O'):
                    self.agent.q_table[(old_state, action)] += 0.5
                elif self.blocks_opponent_win(board, action):
                    self.agent.q_table[(old_state, action)] += 0.3
                elif action == 4:  # Center
                    self.agent.q_table[(old_state, action)] += 0.1
        else:
            # Smarter opponent (minimax depth 1)
            move = self.get_smart_opponent_move(board)
            if move is not None:
                board[move] = 'X'

        ai_turn = not ai_turn
```

De asemenea, modifică parametrii de antrenament:

```python
# Parametri îmbunătățiți
self.ml_agent = QLearningAgent(
    learning_rate=0.1,      # Mai mic pentru stabilitate
    discount_factor=0.95,   # Păstrează
    epsilon=0.2            # Explorare moderată
)

# Antrenament în faze:
# Faza 1: 5000 jocuri cu epsilon=0.5 (multă explorare)
# Faza 2: 5000 jocuri cu epsilon=0.2 (explorare moderată)  
# Faza 3: 5000 jocuri cu epsilon=0.05 (exploatare)
```

ALTE ÎMBUNĂTĂȚIRI RECOMANDATE:

1. **Experience Replay**: Salvează ultimele 1000 jocuri și re-antrenează pe ele periodic
2. **Opponent Pool**: Antrenează împotriva mai multor tipuri de adversari
3. **Reward Shaping**: Recompense imediate pentru mutări strategice
4. **Curriculum Learning**: Începe cu adversari ușori, crește dificultatea gradual
"""

        self.problems_text.append("\n\n" + fixes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    inspector = ModelInspector()
    inspector.show()
    sys.exit(app.exec_())