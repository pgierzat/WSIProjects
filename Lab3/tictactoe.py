class Game:
    def __init__(self, size):
        if size < 3 or size > 9:
            return "size has to be between 3 and 9"
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
                return (True, f"Player{row[0]} wins!")
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


def minimax():
    


def main():
    size = int(input("Enter the size of the board (3-9): "))
    game = Game(size)
    players = ['X', 'O']
    current_player = 0

    while True:
        game.show_board()
        if players[current_player] == 'X':
            print(f"Player {players[current_player]}'s turn")
            row = int(input("Enter row: "))
            col = int(input("Enter column: "))
            game.add_move((row, col), players[current_player])
        else:
            print("AI's turn")
            move = best_move(game._board)
            game.add_move(move, 'O')

        won, message = game.check_state()
        if won:
            game.show_board()
            print(message)
            break
        elif message == "It's a draw!":
            game.show_board()
            print(message)
            break
        current_player = 1 - current_player


if __name__ == "__main__":
    main()