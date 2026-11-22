import chess
import chess.engine

from chess_logic.analyzer import PositionAnalyzer
from chess_logic.fen import board_to_fen


MATE_SCORE = 100000


class StockfishAnalyzer(PositionAnalyzer):
    def __init__(
        self,
        engine_path,
        depth=15
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            str(engine_path)
        )

        self.depth = depth

    def analyze(self, position):
        fen = self._normalize_to_fen(
            position
        )

        board = chess.Board(fen)

        return self._analyse_board(
            board
        )

    def evaluate_move(
        self,
        fen,
        move_uci
    ):
        board = chess.Board(fen)

        move = chess.Move.from_uci(
            move_uci
        )

        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal move {move_uci} "
                f"for position {fen}"
            )

        result = self._analyse_board(
            board,
            root_moves=[move]
        )

        result["forced_move"] = move_uci

        return result

    def _analyse_board(
        self,
        board,
        root_moves=None
    ):
        root_turn = board.turn

        result = self.engine.analyse(
            board,
            chess.engine.Limit(
                depth=self.depth
            ),
            root_moves=root_moves
        )

        white_score = (
            result["score"]
            .pov(chess.WHITE)
        )

        mover_score = (
            result["score"]
            .pov(root_turn)
        )

        score = white_score.score(
            mate_score=MATE_SCORE
        )

        mate = white_score.mate()

        score_for_side_to_move = (
            mover_score.score(
                mate_score=MATE_SCORE
            )
        )

        mate_for_side_to_move = (
            mover_score.mate()
        )

        pv = result.get(
            "pv",
            []
        )

        best_line = [
            move.uci()
            for move in pv
        ]

        if best_line:
            best_move = best_line[0]
        else:
            best_move = None

        return {
            "score": score,
            "mate": mate,

            "score_for_side_to_move":
                score_for_side_to_move,

            "mate_for_side_to_move":
                mate_for_side_to_move,

            "best_move": best_move,
            "best_line": best_line,
            "depth": self.depth
        }

    def _normalize_to_fen(
        self,
        position
    ):
        if isinstance(
            position,
            str
        ):
            return position

        if isinstance(
            position,
            list
        ):
            return board_to_fen(
                position
            )

        if isinstance(
            position,
            dict
        ):
            if "fen" in position:
                return position["fen"]

            if "board" in position:
                return board_to_fen(
                    position["board"]
                )

        raise TypeError(
            "Position must be a FEN string, "
            "8x8 board list, or dict with "
            "'fen'/'board'."
        )

    def close(self):
        self.engine.quit()