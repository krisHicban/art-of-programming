import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class TicTacToe(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tic-Tac-Toe')
        self.setGeometry(100, 100, 300, 400)
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                font-size: 24px;
                font-weight: bold;
                border: 2px solid #2c3e50;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #4a627a;
            }
            #statusLabel {
                font-size: 18px;
                font-weight: bold;
            }
            #newGameButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            #newGameButton:hover {
                background-color: #c0392b;
            }
        """)

        self.board = [''] * 9
        self.buttons = []
        self.player_turn = True

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.status_label = QLabel("Your turn (X)", self)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        grid_layout = QVBoxLayout()
        grid_layout.setSpacing(5)

        for i in range(3):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            for j in range(3):
                button = QPushButton('', self)
                button.setFixedSize(80, 80)
                button.clicked.connect(lambda _, b=button, index=i*3+j: self.on_button_click(b, index))
                self.buttons.append(button)
                row_layout.addWidget(button)
            grid_layout.addLayout(row_layout)
        
        main_layout.addLayout(grid_layout)

        self.new_game_button = QPushButton('New Game', self)
        self.new_game_button.setObjectName("newGameButton")
        self.new_game_button.clicked.connect(self.reset_game)
        main_layout.addWidget(self.new_game_button)

        self.setLayout(main_layout)

    def on_button_click(self, button, index):
        if self.board[index] == '' and self.player_turn:
            self.board[index] = 'X'
            button.setText('X')
            button.setStyleSheet("color: #3498db;")
            self.player_turn = False
            if not self.check_winner():
                self.computer_move()

    def computer_move(self):
        if '' not in self.board:
            return

        # 1. Check if computer can win
        for i in range(9):
            if self.board[i] == '':
                self.board[i] = 'O'
                if self.check_winner(silent=True) == 'O':
                    self.buttons[i].setText('O')
                    self.buttons[i].setStyleSheet("color: #e74c3c;")
                    self.check_winner()
                    return
                self.board[i] = ''

        # 2. Check if player can win and block
        for i in range(9):
            if self.board[i] == '':
                self.board[i] = 'X'
                if self.check_winner(silent=True) == 'X':
                    self.board[i] = 'O'
                    self.buttons[i].setText('O')
                    self.buttons[i].setStyleSheet("color: #e74c3c;")
                    self.player_turn = True
                    self.check_winner()
                    return
                self.board[i] = ''
        
        # 3. Otherwise, take a random spot
        empty_cells = [i for i, val in enumerate(self.board) if val == '']
        if empty_cells:
            move = random.choice(empty_cells)
            self.board[move] = 'O'
            self.buttons[move].setText('O')
            self.buttons[move].setStyleSheet("color: #e74c3c;")

        self.player_turn = True
        self.check_winner()

    def check_winner(self, silent=False):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)             # diagonals
        ]
        for a, b, c in win_conditions:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != '':
                if not silent:
                    self.end_game(self.board[a])
                return self.board[a]
        
        if '' not in self.board:
            if not silent:
                self.end_game('Draw')
            return 'Draw'
        
        if not silent:
            self.status_label.setText("Your turn (X)" if self.player_turn else "Computer's turn (O)")
        return None

    def end_game(self, winner):
        if winner == 'Draw':
            self.status_label.setText("It's a Draw!")
        else:
            self.status_label.setText(f"{winner} wins!")
        
        for button in self.buttons:
            button.setEnabled(False)
        self.player_turn = False

    def reset_game(self):
        self.board = [''] * 9
        self.player_turn = True
        self.status_label.setText("Your turn (X)")
        for button in self.buttons:
            button.setText('')
            button.setEnabled(True)
            button.setStyleSheet("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = TicTacToe()
    game.show()
    sys.exit(app.exec_())