from chess_logic.analyzer import PositionAnalyzer


PIECE_VALUES = {
    "P": 100,
    "N": 320,
    "B": 330,
    "R": 500,
    "Q": 900,
    "K": 0,

    "p": -100,
    "n": -320,
    "b": -330,
    "r": -500,
    "q": -900,
    "k": 0
}


class CustomEvaluator(PositionAnalyzer):
    def analyze(self, board):
        material_score = self.evaluate_material(board)

        total_score = material_score

        return {
            "material": material_score,
            "total": total_score,
            "best_move": None
        }

    def evaluate_material(self, board):
        score = 0

        for row in board:
            for square in row:
                if square is not None:
                    score += PIECE_VALUES[square]

        return score