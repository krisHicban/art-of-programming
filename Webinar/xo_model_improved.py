"""
🧠 X și O cu Q-Learning - VERSIUNE REPARATĂ
============================================
Aceasta este versiunea corectată care rezolvă problemele de antrenament.

Schimbări majore:
1. Sistem de recompense corectat
2. Adversar inteligent (nu random)
3. Reward shaping pentru mutări strategice
4. Antrenament în faze cu epsilon decay

Autor: Lao & Claude
"""

import random
import pickle
from collections import defaultdict
import numpy as np


class ImprovedQLearningAgent:
    """
    🧠 Agent Q-Learning îmbunătățit cu sistem de recompense corectat
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.q_table = defaultdict(float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Tracking pentru debugging
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Memory pentru experience replay
        self.memory = []
        self.max_memory = 1000

    def state_to_string(self, board):
        """Convertește tabla într-un string pentru Q-table"""
        return ''.join(board).replace(' ', '_')

    def get_valid_actions(self, board):
        """Găsește toate pozițiile libere"""
        return [i for i, cell in enumerate(board) if cell == '']

    def check_winner(self, board):
        """Verifică câștigătorul"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rânduri
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # coloane
            [0, 4, 8], [2, 4, 6]  # diagonale
        ]

        for pattern in win_patterns:
            if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != '':
                return board[pattern[0]]
        return None

    def is_winning_move(self, board, position, player):
        """Verifică dacă o mutare câștigă jocul"""
        if board[position] != '':
            return False
        test_board = board[:]
        test_board[position] = player
        return self.check_winner(test_board) == player

    def is_blocking_move(self, board, position, player):
        """Verifică dacă o mutare blochează adversarul"""
        opponent = 'X' if player == 'O' else 'O'
        return self.is_winning_move(board, position, opponent)

    def count_two_in_row(self, board, player):
        """Numără câte linii au 2 piese ale jucătorului"""
        count = 0
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for pattern in win_patterns:
            pieces = [board[i] for i in pattern]
            if pieces.count(player) == 2 and pieces.count('') == 1:
                count += 1
        return count

    def evaluate_board_position(self, board, player):
        """
        Evaluează cât de bună este o poziție pentru un jucător
        Returnează un scor între -1 și 1
        """
        opponent = 'X' if player == 'O' else 'O'

        # Verifică câștig/pierdere
        winner = self.check_winner(board)
        if winner == player:
            return 1.0
        elif winner == opponent:
            return -1.0
        elif '' not in board:
            return 0.0  # Egalitate

        # Evaluare euristică
        score = 0.0

        # Bonus pentru centru
        if board[4] == player:
            score += 0.2
        elif board[4] == opponent:
            score -= 0.2

        # Bonus pentru colțuri
        corners = [0, 2, 6, 8]
        for corner in corners:
            if board[corner] == player:
                score += 0.1
            elif board[corner] == opponent:
                score -= 0.1

        # Bonus pentru potențiale linii câștigătoare
        player_lines = self.count_two_in_row(board, player)
        opponent_lines = self.count_two_in_row(board, opponent)
        score += player_lines * 0.15
        score -= opponent_lines * 0.15

        return max(-1.0, min(1.0, score))  # Clamp între -1 și 1

    def choose_action(self, board, training=True):
        """
        Alege o acțiune folosind epsilon-greedy cu evaluare strategică
        """
        state = self.state_to_string(board)
        valid_actions = self.get_valid_actions(board)

        if not valid_actions:
            return None

        # În modul non-training, folosește mai puțină explorare
        epsilon = self.epsilon if training else max(0.05, self.epsilon * 0.1)

        if random.random() < epsilon:
            # Explorare - dar cu o preferință pentru mutări strategice
            weights = []
            for action in valid_actions:
                weight = 1.0
                if self.is_winning_move(board, action, 'O'):
                    weight = 10.0  # Preferă puternic mutările câștigătoare
                elif self.is_blocking_move(board, action, 'O'):
                    weight = 5.0  # Preferă blocarea
                elif action == 4 and board[4] == '':
                    weight = 3.0  # Preferă centrul
                elif action in [0, 2, 6, 8]:
                    weight = 2.0  # Preferă colțurile
                weights.append(weight)

            # Alege random dar cu pondere
            total = sum(weights)
            weights = [w / total for w in weights]
            return np.random.choice(valid_actions, p=weights)
        else:
            # Exploatare - folosește Q-values + evaluare euristică
            action_values = {}

            for action in valid_actions:
                # Q-value din tabel
                q_value = self.q_table.get((state, action), 0.0)

                # Bonus euristic pentru evaluare imediată
                test_board = board[:]
                test_board[action] = 'O'
                heuristic = self.evaluate_board_position(test_board, 'O')

                # Combinație între Q-value învățat și euristică
                action_values[action] = q_value + 0.1 * heuristic

            # Alege acțiunea cu valoarea maximă
            max_value = max(action_values.values())
            best_actions = [a for a, v in action_values.items() if v == max_value]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state, done=False):
        """
        Actualizează Q-value folosind formula Q-learning standard
        """
        current_q = self.q_table.get((state, action), 0.0)

        if done:
            # Stare terminală - nu mai există stare următoare
            target = reward
        else:
            # Găsește valoarea maximă Q pentru starea următoare
            next_valid_actions = self.get_valid_actions(self.string_to_board(next_state))
            if next_valid_actions:
                max_next_q = max(
                    self.q_table.get((next_state, a), 0.0)
                    for a in next_valid_actions
                )
            else:
                max_next_q = 0.0

            target = reward + self.discount_factor * max_next_q

        # Actualizare Q-value
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[(state, action)] = new_q

        # Salvează în memorie pentru experience replay
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def string_to_board(self, state_string):
        """Convertește string-ul înapoi în board list"""
        return [c if c != '_' else '' for c in state_string]

    def experience_replay(self, batch_size=32):
        """
        Re-antrenează pe experiențe anterioare din memorie
        """
        if len(self.memory) < batch_size:
            return

        # Sample random din memorie
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            current_q = self.q_table.get((state, action), 0.0)

            if done:
                target = reward
            else:
                next_board = self.string_to_board(next_state)
                next_valid_actions = self.get_valid_actions(next_board)
                if next_valid_actions:
                    max_next_q = max(
                        self.q_table.get((next_state, a), 0.0)
                        for a in next_valid_actions
                    )
                else:
                    max_next_q = 0.0
                target = reward + self.discount_factor * max_next_q

            # Update cu learning rate mai mic pentru stabilitate
            new_q = current_q + self.learning_rate * 0.5 * (target - current_q)
            self.q_table[(state, action)] = new_q

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Reduce epsilon gradual pentru mai puțină explorare în timp"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def reset_epsilon(self):
        """Resetează epsilon la valoarea inițială"""
        self.epsilon = self.initial_epsilon

    def get_statistics(self):
        """Returnează statisticile de antrenament"""
        if self.total_games == 0:
            return {
                "total": 0, "wins": 0, "losses": 0, "draws": 0,
                "win_rate": 0, "loss_rate": 0, "draw_rate": 0
            }

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
        """Salvează modelul antrenat"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'stats': self.get_statistics(),
                'params': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.epsilon
                },
                'memory': self.memory[-100:]  # Salvează ultimele 100 experiențe
            }, f)

    def load_model(self, filename):
        """Încarcă un model salvat"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                stats = data['stats']
                self.total_games = stats['total']
                self.wins = stats['wins']
                self.losses = stats['losses']
                self.draws = stats['draws']
                if 'memory' in data:
                    self.memory = data['memory']
                return True
        except FileNotFoundError:
            return False


class SmartOpponent:
    """
    🤖 Adversar inteligent pentru antrenament mai bun
    Folosește minimax cu adâncime limitată
    """

    def __init__(self, difficulty=0.7):
        self.difficulty = difficulty  # 0.0 = random, 1.0 = perfect

    def check_winner(self, board):
        """Verifică câștigătorul"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for pattern in win_patterns:
            if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != '':
                return board[pattern[0]]
        return None

    def minimax(self, board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        """
        Algoritm minimax cu alpha-beta pruning
        """
        winner = self.check_winner(board)

        # Stări terminale
        if winner == 'X':  # Opponent (maximizing) wins
            return 1
        elif winner == 'O':  # AI (minimizing) loses
            return -1
        elif '' not in board:  # Draw
            return 0

        if depth == 0:
            return 0  # Evaluare simplă la adâncime 0

        if is_maximizing:
            max_eval = -float('inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = 'X'
                    eval_score = self.minimax(board, depth - 1, False, alpha, beta)
                    board[i] = ''
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = 'O'
                    eval_score = self.minimax(board, depth - 1, True, alpha, beta)
                    board[i] = ''
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval

    def get_best_move(self, board):
        """Găsește cea mai bună mutare folosind minimax"""
        best_move = None
        best_value = -float('inf')

        # Verifică dacă poate câștiga imediat
        for i in range(9):
            if board[i] == '':
                board[i] = 'X'
                if self.check_winner(board) == 'X':
                    board[i] = ''
                    return i
                board[i] = ''

        # Verifică dacă trebuie să blocheze
        for i in range(9):
            if board[i] == '':
                board[i] = 'O'
                if self.check_winner(board) == 'O':
                    board[i] = ''
                    return i
                board[i] = ''

        # Folosește minimax pentru celelalte cazuri
        for i in range(9):
            if board[i] == '':
                board[i] = 'X'
                move_value = self.minimax(board, 3, False)  # Depth 3 pentru viteză
                board[i] = ''

                if move_value > best_value:
                    best_value = move_value
                    best_move = i

        return best_move

    def choose_move(self, board):
        """
        Alege o mutare bazată pe dificultate
        """
        if random.random() < self.difficulty:
            # Mutare inteligentă
            return self.get_best_move(board)
        else:
            # Mutare random
            valid_moves = [i for i in range(9) if board[i] == '']
            return random.choice(valid_moves) if valid_moves else None


class ImprovedTrainer:
    """
    🎓 Sistem de antrenament îmbunătățit cu curriculum learning
    """

    def __init__(self, agent):
        self.agent = agent
        self.opponents = [
            SmartOpponent(difficulty=0.0),  # Random
            SmartOpponent(difficulty=0.3),  # Easy
            SmartOpponent(difficulty=0.5),  # Medium
            SmartOpponent(difficulty=0.7),  # Hard
            SmartOpponent(difficulty=0.9),  # Very Hard
            SmartOpponent(difficulty=1.0),  # Perfect
        ]

    def calculate_reward(self, board, action, player, game_result=None):
        """
        Calculează recompensa pentru o mutare
        """
        if game_result is not None:
            # Recompense finale
            if game_result == player:
                return 1.0  # Victorie
            elif game_result == 'draw':
                return 0.1  # Egalitate - mică recompensă pozitivă
            else:
                return -1.0  # Înfrângere

        # Recompense intermediare (reward shaping)
        reward = 0.0

        # Verifică dacă mutarea câștigă
        if self.agent.is_winning_move(board, action, player):
            reward += 0.5

        # Verifică dacă mutarea blochează
        if self.agent.is_blocking_move(board, action, player):
            reward += 0.3

        # Bonus pentru centru
        if action == 4:
            reward += 0.05

        # Bonus pentru colțuri
        if action in [0, 2, 6, 8]:
            reward += 0.02

        return reward

    def play_training_game(self, opponent, verbose=False):
        """
        Joacă un joc de antrenament cu recompense corecte
        """
        board = [''] * 9
        game_history = []

        # Decide cine începe
        ai_starts = random.choice([True, False])
        current_player = 'O' if ai_starts else 'X'

        while True:
            # Verifică sfârșitul jocului
            winner = self.agent.check_winner(board)
            if winner is not None or '' not in board:
                # Jocul s-a terminat
                if winner == 'O':
                    self.agent.wins += 1
                    final_result = 'O'
                elif winner == 'X':
                    self.agent.losses += 1
                    final_result = 'X'
                else:
                    self.agent.draws += 1
                    final_result = 'draw'

                self.agent.total_games += 1

                # Actualizează Q-values pentru toate mutările din istoric
                # Folosește temporal difference learning
                for i in range(len(game_history)):
                    state, action, _ = game_history[i]

                    if i == len(game_history) - 1:
                        # Ultima mutare - recompensă finală
                        reward = self.calculate_reward(board, action, 'O', final_result)
                        self.agent.update_q_value(state, action, reward, state, done=True)
                    else:
                        # Mutări intermediare
                        next_state = game_history[i + 1][0] if i + 1 < len(game_history) else state
                        reward = self.calculate_reward(board, action, 'O')
                        self.agent.update_q_value(state, action, reward, next_state, done=False)

                if verbose:
                    print(f"Game ended: {final_result}")
                    self.print_board(board)

                break

            if current_player == 'O':
                # Rândul AI-ului
                state = self.agent.state_to_string(board)
                action = self.agent.choose_action(board, training=True)

                if action is not None and board[action] == '':
                    board[action] = 'O'
                    game_history.append((state, action, board[:]))
                    current_player = 'X'
                else:
                    # Mutare invalidă - penalizare și sfârșit joc
                    if action is not None:
                        self.agent.update_q_value(state, action, -0.5, state, done=True)
                    break
            else:
                # Rândul adversarului
                move = opponent.choose_move(board)
                if move is not None and board[move] == '':
                    board[move] = 'X'
                    current_player = 'O'

    def print_board(self, board):
        """Afișează tabla de joc"""
        for i in range(0, 9, 3):
            row = board[i:i + 3]
            print(' | '.join(c if c else ' ' for c in row))
            if i < 6:
                print('---------')

    def train_curriculum(self, episodes_per_level=1000, verbose=True):
        """
        Antrenament cu curriculum learning - începe ușor, crește dificultatea
        """
        total_episodes = episodes_per_level * len(self.opponents)
        current_episode = 0

        for level, opponent in enumerate(self.opponents):
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"Level {level + 1}: Difficulty {opponent.difficulty:.1f}")
                print(f"{'=' * 50}")

            # Resetează epsilon pentru explorare la început de nivel
            if level > 0:
                self.agent.epsilon = max(0.2, self.agent.initial_epsilon * (0.7 ** level))

            level_wins = 0
            level_losses = 0
            level_draws = 0

            for episode in range(episodes_per_level):
                # Joacă un joc
                wins_before = self.agent.wins
                losses_before = self.agent.losses
                draws_before = self.agent.draws

                self.play_training_game(opponent, verbose=False)

                # Track level statistics
                if self.agent.wins > wins_before:
                    level_wins += 1
                elif self.agent.losses > losses_before:
                    level_losses += 1
                else:
                    level_draws += 1

                # Experience replay la fiecare 10 jocuri
                if episode % 10 == 0 and episode > 0:
                    self.agent.experience_replay()

                # Decay epsilon
                self.agent.decay_epsilon(decay_rate=0.999)

                # Progress report
                if verbose and episode % 100 == 0 and episode > 0:
                    win_rate = level_wins / (episode + 1)
                    print(f"  Episode {episode}: Win rate {win_rate:.1%}")

                current_episode += 1

            # Level complete statistics
            if verbose:
                total_level_games = level_wins + level_losses + level_draws
                print(f"\nLevel {level + 1} Complete!")
                print(f"  Wins: {level_wins} ({level_wins / total_level_games:.1%})")
                print(f"  Losses: {level_losses} ({level_losses / total_level_games:.1%})")
                print(f"  Draws: {level_draws} ({level_draws / total_level_games:.1%})")
                print(f"  Q-Table size: {len(self.agent.q_table)}")
                print(f"  Current epsilon: {self.agent.epsilon:.3f}")

    def train_self_play(self, episodes=1000, verbose=True):
        """
        Self-play training - AI-ul joacă împotriva versiunilor anterioare
        """
        # Salvează o copie a Q-table-ului curent pentru adversar
        opponent_q_table = dict(self.agent.q_table)

        for episode in range(episodes):
            board = [''] * 9
            game_history = []

            # Alternează cine începe
            current_player = 'O' if episode % 2 == 0 else 'X'

            while True:
                winner = self.agent.check_winner(board)
                if winner is not None or '' not in board:
                    # Procesează sfârșitul jocului ca în curriculum training
                    if winner == 'O':
                        self.agent.wins += 1
                        final_result = 'O'
                    elif winner == 'X':
                        self.agent.losses += 1
                        final_result = 'X'
                    else:
                        self.agent.draws += 1
                        final_result = 'draw'

                    self.agent.total_games += 1

                    # Actualizează Q-values
                    for i in range(len(game_history)):
                        state, action, player = game_history[i]

                        if player == 'O':  # Doar pentru mutările AI-ului nostru
                            if i == len(game_history) - 1:
                                reward = self.calculate_reward(board, action, 'O', final_result)
                                self.agent.update_q_value(state, action, reward, state, done=True)
                            else:
                                # Găsește următoarea stare a AI-ului nostru
                                next_state = None
                                for j in range(i + 1, len(game_history)):
                                    if game_history[j][2] == 'O':
                                        next_state = game_history[j][0]
                                        break

                                if next_state:
                                    reward = self.calculate_reward(board, action, 'O')
                                    self.agent.update_q_value(state, action, reward, next_state, done=False)
                    break

                state = self.agent.state_to_string(board)

                if current_player == 'O':
                    # AI-ul nostru
                    action = self.agent.choose_action(board, training=True)
                    if action is not None and board[action] == '':
                        board[action] = 'O'
                        game_history.append((state, action, 'O'))
                        current_player = 'X'
                else:
                    # Adversar (versiune anterioară)
                    valid_actions = [i for i in range(9) if board[i] == '']
                    if valid_actions:
                        # Folosește Q-table-ul salvat
                        action_values = {
                            a: opponent_q_table.get((state, a), 0.0)
                            for a in valid_actions
                        }
                        max_value = max(action_values.values())
                        best_actions = [a for a, v in action_values.items() if v == max_value]
                        action = random.choice(best_actions)

                        board[action] = 'X'
                        game_history.append((state, action, 'X'))
                        current_player = 'O'

            # Actualizează adversarul periodic
            if episode % 200 == 0 and episode > 0:
                opponent_q_table = dict(self.agent.q_table)
                if verbose:
                    stats = self.agent.get_statistics()
                    print(f"Episode {episode}: Win rate {stats['win_rate']:.1%}")

            # Experience replay
            if episode % 10 == 0:
                self.agent.experience_replay()

            # Decay epsilon
            self.agent.decay_epsilon()


# Exemplu de utilizare
if __name__ == "__main__":
    print("🧠 Antrenament Q-Learning Îmbunătățit pentru X și O")
    print("=" * 50)

    # Creează agent nou
    agent = ImprovedQLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.5  # Începe cu explorare mare
    )

    # Creează trainer
    trainer = ImprovedTrainer(agent)

    print("\n📚 Faza 1: Curriculum Learning (6000 jocuri)")
    print("-" * 50)
    trainer.train_curriculum(episodes_per_level=1000, verbose=True)

    print("\n🤖 Faza 2: Self-Play (2000 jocuri)")
    print("-" * 50)
    trainer.train_self_play(episodes=2000, verbose=True)

    # Statistici finale
    stats = agent.get_statistics()
    print("\n📊 STATISTICI FINALE")
    print("=" * 50)
    print(f"Total jocuri: {stats['total']}")
    print(f"Victorii: {stats['wins']} ({stats['win_rate']:.1%})")
    print(f"Înfrângeri: {stats['losses']} ({stats['loss_rate']:.1%})")
    print(f"Egalități: {stats['draws']} ({stats['draw_rate']:.1%})")
    print(f"Q-Table entries: {len(agent.q_table)}")
    print(f"Final epsilon: {agent.epsilon:.3f}")

    # Salvează modelul
    agent.save_model('xo_ai_model_improved.pkl')
    print("\n💾 Model salvat ca 'xo_ai_model_improved.pkl'")

    # Test: joacă un joc demonstrativ
    print("\n🎮 JOC DEMONSTRATIV")
    print("=" * 50)

    board = [''] * 9
    agent.epsilon = 0  # Nu mai explorează în jocul demo
    current_player = 'X'  # Jucătorul uman începe


    def print_demo_board(b):
        print("\n   1 | 2 | 3")
        print("  -----------")
        print(f"   {b[0] if b[0] else '1'} | {b[1] if b[1] else '2'} | {b[2] if b[2] else '3'}")
        print("  -----------")
        print(f"   {b[3] if b[3] else '4'} | {b[4] if b[4] else '5'} | {b[5] if b[5] else '6'}")
        print("  -----------")
        print(f"   {b[6] if b[6] else '7'} | {b[7] if b[7] else '8'} | {b[8] if b[8] else '9'}")
        print()


    print("Tu ești X, AI-ul este O")
    print("Alege o poziție (1-9):")
    print_demo_board(board)

    # Simulare joc automat pentru demonstrație
    demo_moves = [4, 0, 2, 6, 8]  # Mutări simulate
    move_idx = 0

    while True:
        winner = agent.check_winner(board)
        if winner:
            print_demo_board(board)
            if winner == 'X':
                print("🎉 Jucătorul a câștigat!")
            else:
                print("🤖 AI-ul a câștigat!")
            break
        elif '' not in board:
            print_demo_board(board)
            print("🤝 Egalitate!")
            break

        if current_player == 'X':
            # Simulare mutare jucător
            if move_idx < len(demo_moves):
                pos = demo_moves[move_idx]
                move_idx += 1
                if board[pos] == '':
                    board[pos] = 'X'
                    print(f"Jucătorul alege poziția {pos + 1}")
                    current_player = 'O'
        else:
            # AI move
            action = agent.choose_action(board, training=False)
            if action is not None:
                board[action] = 'O'
                print(f"AI-ul alege poziția {action + 1}")

                # Arată de ce a ales această mutare
                state = agent.state_to_string(board)
                q_val = agent.q_table.get((state, action), 0)
                print(f"  (Q-value: {q_val:.3f})")

                current_player = 'X'

    print("\n✅ Antrenament complet!")