import random
import matplotlib.pyplot as plt
import time
import numpy as np



class Game:
    def __init__(self, size):
        if size < 3 or size > 9:
            raise ValueError("size has to be between 3 and 9")
        self._size = size
        self._board = self.create_board()

    def get_size(self):
        return self._size

    def get_board(self):
        return self._board

    def create_board(self):
        N = self.get_size()
        board = [['-' for _ in range(N)] for _ in range(N)]
        return board

    def show_board(self):
        board = self.get_board()
        N = self.get_size()
        for i in range(N):
            row = '   |   '.join(str(cell) for cell in board[i])
            print(row)
            if i < N - 1:
                print('-' * (8 * N - 6))

    def is_empty(self, row, col):
        return self.get_board()[row][col] == "-"

    def add_move(self, move, player):
        row, col = move
        while not self.is_empty(row, col):
            print("Occupied, try again")
            row = int(input("Enter row: "))
            col = int(input("Enter column: "))
        self.get_board()[row][col] = player

    def check_state(self):
        board = self.get_board()
        N = self.get_size()
        for row in board:
            if all(cell == row[0] and cell != '-' for cell in row):
                return (True, f"Player {row[0]} wins!")
        for col in range(N):
            if all(board[row][col] == board[0][col] and board[row][col] != '-' for row in range(N)):
                return (True, f"Player {board[0][col]} wins!")
        if all(board[i][i] == board[0][0] and board[i][i] != '-' for i in range(N)):
            return (True, f"Player {board[0][0]} wins!")
        if all(board[i][N - 1 - i] == board[0][N - 1] and board[i][N - 1 - i] != '-' for i in range(N)):
            return (True, f"Player {board[0][N - 1]} wins!")
        if all(cell != '-' for row in board for cell in row):
            return (True, "It's a draw!")

        return False, "continue"

    def minimax(self, depth, alpha, beta, maximizing_player, current_player, opponent):
        """
        Algorytm minimax z obcinaniem alfa-beta.
        """
        state, result = self.check_state()
        if state:
            if "wins" in result and current_player in result:
                return 10
            elif "wins" in result and opponent in result:
                return -10
            else:  # Remis
                return 0

        if depth == 0:
            return self.heuristic(current_player, opponent)

        if maximizing_player:
            max_eval = -float('inf')
            for i in range(self.get_size()):
                for j in range(self.get_size()):
                    if self.is_empty(i, j):
                        self.get_board()[i][j] = current_player
                        eval = self.minimax(depth - 1, alpha, beta, False, current_player, opponent)
                        self.get_board()[i][j] = '-'
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(self.get_size()):
                for j in range(self.get_size()):
                    if self.is_empty(i, j):
                        self.get_board()[i][j] = opponent
                        eval = self.minimax(depth - 1, alpha, beta, True, current_player, opponent)
                        self.get_board()[i][j] = '-'
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

    def heuristic(self, current_player, opponent):
        N = self.get_size()
        weights = [[3 if (i == 0 or i == N - 1) and (j == 0 or j == N - 1) else
                    4 if i == N // 2 and j == N // 2 else 2 for j in range(N)] for i in range(N)]
        score = 0
        for i in range(N):
            for j in range(N):
                if self.get_board()[i][j] == current_player:
                    score += weights[i][j]
                elif self.get_board()[i][j] == opponent:
                    score -= weights[i][j]
        return score

    def best_move(self, depth, current_player, opponent):
        best_value = -float('inf')
        best_moves = []

        for i in range(self.get_size()):
            for j in range(self.get_size()):
                if self.is_empty(i, j):
                    self.get_board()[i][j] = current_player
                    move_value = self.minimax(depth - 1, -float('inf'), float('inf'), False, current_player, opponent)
                    self.get_board()[i][j] = '-'
                    if move_value > best_value:
                        best_value = move_value
                        best_moves = [(i, j)]
                    elif move_value == best_value:
                        best_moves.append((i, j))
        return random.choice(best_moves) if best_moves else None


def plot_decision_quality(game_class, size, max_depth, simulations=10):
    depths = range(1, max_depth + 1)
    avg_scores = []

    for depth in depths:
        total_score = 0
        for _ in range(simulations):
            game = game_class(size)
            best_move = game.best_move(depth, "X", "O")
            if best_move:
                game.add_move(best_move, "X")
            _, result = game.check_state()
            if "X wins" in result:
                total_score += 1
            elif "O wins" in result:
                total_score -= 1
        avg_scores.append(total_score / simulations)

    plt.figure(figsize=(8, 5))
    plt.plot(depths, avg_scores, marker='o', label='Średnia jakość decyzji')
    plt.title('Jakość decyzji w zależności od głębokości przeszukiwania dla 100 symulacji')
    plt.xlabel('Głębokość przeszukiwania')
    plt.ylabel('Średnia punktacja decyzji')
    plt.grid(True)
    plt.legend()
    plt.savefig("decisionquality.png")


def plot_game_outcomes(game_class, size, max_depth, simulations=10):
    depths = range(1, max_depth + 1)
    outcomes = {"Player X": [], "Player O": [], "Draws": []}

    for depth in depths:
        x_wins = 0
        o_wins = 0
        draws = 0

        for _ in range(simulations):
            game = game_class(size)
            current_player = "X"
            opponent = "O"

            while True:
                best_move = game.best_move(depth, current_player, opponent)
                if best_move:
                    game.add_move(best_move, current_player)
                else:
                    break
                state, result = game.check_state()
                if state:
                    if "X wins" in result:
                        x_wins += 1
                    elif "O wins" in result:
                        o_wins += 1
                    else:
                        draws += 1
                    break
                current_player, opponent = opponent, current_player

        outcomes["Player X"].append(x_wins)
        outcomes["Player O"].append(o_wins)
        outcomes["Draws"].append(draws)

    bar_width = 0.25
    x_indices = np.arange(len(depths))

    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - bar_width, outcomes["Player X"], width=bar_width, label="Player X Wins", color="blue")
    plt.bar(x_indices, outcomes["Player O"], width=bar_width, label="Player O Wins", color="red")
    plt.bar(x_indices + bar_width, outcomes["Draws"], width=bar_width, label="Draws", color="green")

    plt.title(f'Wyniki gier dla różnych głębokości przeszukiwania (Plansza {size}x{size})')
    plt.xlabel('Głębokość przeszukiwania')
    plt.ylabel('Liczba gier')
    plt.xticks(x_indices, depths)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig("game_outcome_6.png")


def plot_best_move_time(game_class, size, max_depth):
    depths = range(1, max_depth + 1)
    times = []

    for depth in depths:
        game = game_class(size)
        start_time = time.time()
        game.best_move(depth, "X", "O")
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    plt.figure(figsize=(10, 6))
    plt.plot(depths, times, marker='o', color='blue', label='Czas znajdowania ruchu')
    plt.title(f'Czas znajdowania najlepszego ruchu (Plansza {size}x{size})')
    plt.xlabel('Głębokość przeszukiwania')
    plt.ylabel('Czas (sekundy)')
    plt.grid(True)
    plt.legend()
    plt.savefig("best_move_time9.png")


def play_game_ai_vs_ai(size, depth=5):
    game = Game(size)
    current_player = "X"
    opponent = "O"

    while True:
        game.show_board()
        print(f"Ruch gracza {current_player}:")

        best_move = game.best_move(depth, current_player, opponent)
        if best_move:
            game.add_move(best_move, current_player)
        else:
            print(f"Gracz {current_player} nie może wykonać ruchu!")
        state, message = game.check_state()
        if state:
            game.show_board()
            print(message)
            break
        current_player, opponent = opponent, current_player


# play_game_ai_vs_ai(3, 2)

# plot_game_outcomes(Game, size=6, max_depth=3, simulations=10)

plot_best_move_time(Game, size=5, max_depth=6)
