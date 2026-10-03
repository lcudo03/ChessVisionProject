import chess
import chess.engine

from chess_logic.analyzer import PositionAnalyzer
from chess_logic.fen import board_to_fen


class StockfishAnalyzer(PositionAnalyzer):
    def __init__(self, engine_path):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def analyze(self, board):
        fen = board_to_fen(board)

        chess_board = chess.Board(fen)

        result = self.engine.analyse(
            chess_board,
            chess.engine.Limit(depth=15)
        )

        score = result["score"].pov(chess.WHITE)
        best_move = result["pv"][0]

        return {
            "score": score,
            "best_move": str(best_move)
        }

    def close(self):
        self.engine.quit()