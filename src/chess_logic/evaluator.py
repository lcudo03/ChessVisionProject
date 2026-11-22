import chess

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
    "k": 0,
}

PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [20, 25, 30, 35, 35, 30, 25, 20],
    [10, 15, 20, 30, 30, 20, 15, 10],
    [5, 10, 15, 25, 25, 15, 10, 5],
    [5, 5, 10, 15, 15, 10, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 10, 20, 30, 30, 20, 10, -30],
    [-30, 10, 20, 30, 30, 20, 10, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]

BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 0, 10, 15, 15, 10, 0, -10],
    [-10, 5, 10, 15, 15, 10, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]

ROOK_TABLE = [
    [0, 0, 5, 10, 10, 5, 0, 0],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [0, 5, 5, 10, 10, 5, 5, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
]

QUEEN_TABLE = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 5, 0, 0, 5, 0, -10],
    [-10, 5, 5, 5, 5, 5, 5, -10],
    [-5, 0, 5, 10, 10, 5, 0, -5],
    [-5, 0, 5, 10, 10, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 5, -10],
    [-10, 0, 5, 0, 0, 5, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]

KING_MIDDLEGAME_TABLE = [
    [-40, -40, -50, -60, -60, -50, -40, -40],
    [-30, -30, -40, -50, -50, -40, -30, -30],
    [-20, -20, -30, -40, -40, -30, -20, -20],
    [-20, -20, -30, -40, -40, -30, -20, -20],
    [-10, -10, -20, -30, -30, -20, -10, -10],
    [0, 0, -10, -20, -20, -10, 0, 0],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [30, 40, 20, 0, 0, 20, 40, 30],
]

KING_ENDGAME_TABLE = [
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-20, -10, 0, 10, 10, 0, -10, -20],
    [-10, 0, 10, 20, 20, 10, 0, -10],
    [0, 10, 20, 30, 30, 20, 10, 0],
    [0, 10, 20, 30, 30, 20, 10, 0],
    [-10, 0, 10, 20, 20, 10, 0, -10],
    [-20, -10, 0, 10, 10, 0, -10, -20],
    [-30, -20, -10, 0, 0, -10, -20, -30],
]

PIECE_SQUARE_TABLES = {
    "P": PAWN_TABLE,
    "N": KNIGHT_TABLE,
    "B": BISHOP_TABLE,
    "R": ROOK_TABLE,
    "Q": QUEEN_TABLE,
}

PASSED_PAWN_BONUSES = [
    0,
    5,
    10,
    20,
    35,
    60,
    100,
]

MATE_SCORE = 100000
SEARCH_DEPTH = 4
QUIESCENCE_DEPTH = 4

TT_EXACT = "EXACT"
TT_LOWERBOUND = "LOWERBOUND"
TT_UPPERBOUND = "UPPERBOUND"


class CustomEvaluator(PositionAnalyzer):
    def __init__(self):
        super().__init__()

        self.transposition_table = {}
        self.evaluation_cache = {}

        self.tt_hits = 0
        self.tt_stores = 0
        self.evaluation_cache_hits = 0

    def analyze(self, position):
        evaluation = self.evaluate(position)

        fen = self._extract_fen(position)

        best_move = None
        search_evaluation = None
        best_line = []

        if fen is not None:
            (
                best_move,
                search_evaluation,
                best_line
            ) = self.search_best_move(
                fen,
                depth=SEARCH_DEPTH,
            )

        return {
            **evaluation,
            "best_move": best_move,
            "search_evaluation": search_evaluation,
            "best_line": best_line,
        }

    def evaluate(self, position):
        board = self._normalize_position(position)
        fen = self._extract_fen(position)

        if fen is not None:
            cached = self.evaluation_cache.get(
                fen
            )

            if cached is not None:
                self.evaluation_cache_hits += 1
                return cached.copy()

            terminal_score = (
                self.evaluate_terminal_position(
                    fen
                )
            )

            if terminal_score is not None:
                result = {
                    "material": 0,
                    "position": 0,
                    "mobility": 0,
                    "pawn_structure": 0,
                    "king_safety": 0,
                    "strategic": 0,
                    "total": terminal_score,
                }

                self.evaluation_cache[
                    fen
                ] = result

                return result.copy()

        material_score = (
            self.evaluate_material(
                board
            )
        )

        positional_score = (
            self.evaluate_position(
                board
            )
        )

        pawn_structure_score = (
            self.evaluate_pawn_structure(
                board
            )
        )

        king_safety_score = (
            self.evaluate_king_safety(
                board
            )
        )

        strategic_score = (
            self.evaluate_strategic_features(
                board
            )
        )

        if fen is not None:
            mobility_score = (
                self.evaluate_mobility(
                    fen
                )
            )
        else:
            mobility_score = 0

        total_score = (
            material_score
            + positional_score
            + mobility_score
            + pawn_structure_score
            + king_safety_score
            + strategic_score
        )

        result = {
            "material": material_score,
            "position": positional_score,
            "mobility": mobility_score,
            "pawn_structure": pawn_structure_score,
            "king_safety": king_safety_score,
            "strategic": strategic_score,
            "total": total_score,
        }

        if fen is not None:
            self.evaluation_cache[
                fen
            ] = result

        return result.copy()

    def find_best_move(
        self,
        fen,
        depth=2
    ):
        best_move, _, _ = (
            self.search_best_move(
                fen,
                depth
            )
        )

        return best_move

    def search_best_move(
        self,
        fen,
        depth=2
    ):
        self._reset_search_state()

        board = chess.Board(
            fen
        )

        if board.is_game_over():
            evaluation = self.evaluate(
                fen
            )

            return (
                None,
                evaluation["total"],
                []
            )

        root_turn = board.turn

        color = (
            1
            if root_turn == chess.WHITE
            else -1
        )

        best_score = float(
            "-inf"
        )

        best_move = None
        best_line = []

        alpha = float(
            "-inf"
        )

        beta = float(
            "inf"
        )

        moves = self.order_moves(
            board
        )

        for move in moves:
            board.push(
                move
            )

            (
                child_score,
                child_line
            ) = self.negamax(
                board=board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                color=-color,
            )

            score = -child_score

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

                best_line = [
                    move.uci()
                ] + child_line

            alpha = max(
                alpha,
                score
            )

        if best_move is None:
            return (
                None,
                None,
                []
            )

        if root_turn == chess.WHITE:
            white_perspective_score = (
                best_score
            )
        else:
            white_perspective_score = (
                -best_score
            )

        return (
            best_move.uci(),
            white_perspective_score,
            best_line,
        )

    def negamax(
        self,
        board,
        depth,
        alpha,
        beta,
        color
    ):
        if board.is_game_over():
            evaluation = self.evaluate(
                board.fen()
            )

            return (
                color
                * evaluation["total"],
                []
            )

        if depth == 0:
            return self.quiescence(
                board=board,
                alpha=alpha,
                beta=beta,
                color=color,
                depth=QUIESCENCE_DEPTH,
            )

        alpha_original = alpha
        beta_original = beta

        key = self._transposition_key(
            board
        )

        entry = (
            self.transposition_table.get(
                key
            )
        )

        preferred_move = None

        if entry is not None:
            preferred_move = entry.get(
                "best_move"
            )

            if entry["depth"] >= depth:
                self.tt_hits += 1

                if (
                    entry["flag"]
                    == TT_EXACT
                ):
                    return (
                        entry["score"],
                        entry["line"].copy(),
                    )

                if (
                    entry["flag"]
                    == TT_LOWERBOUND
                ):
                    alpha = max(
                        alpha,
                        entry["score"],
                    )

                elif (
                    entry["flag"]
                    == TT_UPPERBOUND
                ):
                    beta = min(
                        beta,
                        entry["score"],
                    )

                if alpha >= beta:
                    return (
                        entry["score"],
                        entry["line"].copy(),
                    )

        best_score = float(
            "-inf"
        )

        best_line = []
        best_move = None

        moves = self.order_moves(
            board,
            preferred_move=preferred_move,
        )

        for move in moves:
            board.push(
                move
            )

            (
                child_score,
                child_line
            ) = self.negamax(
                board=board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                color=-color,
            )

            score = -child_score

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move.uci()

                best_line = [
                    move.uci()
                ] + child_line

            alpha = max(
                alpha,
                score
            )

            if alpha >= beta:
                break

        if best_score <= alpha_original:
            flag = TT_UPPERBOUND

        elif best_score >= beta_original:
            flag = TT_LOWERBOUND

        else:
            flag = TT_EXACT

        current_entry = (
            self.transposition_table.get(
                key
            )
        )

        if (
            current_entry is None
            or depth >= current_entry["depth"]
        ):
            self.transposition_table[
                key
            ] = {
                "depth": depth,
                "score": best_score,
                "flag": flag,
                "best_move": best_move,
                "line": best_line.copy(),
            }

            self.tt_stores += 1

        return (
            best_score,
            best_line,
        )

    def quiescence(
        self,
        board,
        alpha,
        beta,
        color,
        depth
    ):
        if board.is_game_over():
            evaluation = self.evaluate(
                board.fen()
            )

            return (
                color
                * evaluation["total"],
                []
            )

        evaluation = self.evaluate(
            board.fen()
        )

        stand_pat = (
            color
            * evaluation["total"]
        )

        if depth <= 0:
            return (
                stand_pat,
                []
            )

        in_check = board.is_check()

        if not in_check:
            if stand_pat >= beta:
                return (
                    stand_pat,
                    []
                )

            if stand_pat > alpha:
                alpha = stand_pat

        if in_check:
            moves = list(
                board.legal_moves
            )

        else:
            moves = [
                move
                for move in board.legal_moves
                if (
                    board.is_capture(
                        move
                    )
                    or move.promotion
                    is not None
                )
            ]

        if not moves:
            return (
                stand_pat,
                []
            )

        moves = self.order_moves(
            board,
            moves=moves,
        )

        best_line = []

        for move in moves:
            board.push(
                move
            )

            (
                child_score,
                child_line
            ) = self.quiescence(
                board=board,
                alpha=-beta,
                beta=-alpha,
                color=-color,
                depth=depth - 1,
            )

            score = -child_score

            board.pop()

            if score >= beta:
                return (
                    score,
                    [
                        move.uci()
                    ] + child_line
                )

            if score > alpha:
                alpha = score

                best_line = [
                    move.uci()
                ] + child_line

        return (
            alpha,
            best_line,
        )

    def order_moves(
        self,
        board,
        moves=None,
        preferred_move=None
    ):
        if moves is None:
            moves = list(
                board.legal_moves
            )
        else:
            moves = list(
                moves
            )

        def move_score(move):
            score = 0

            if (
                preferred_move
                is not None
                and move.uci()
                == preferred_move
            ):
                score += 20000

            if board.is_capture(
                move
            ):
                score += 1000

                victim = board.piece_at(
                    move.to_square
                )

                attacker = board.piece_at(
                    move.from_square
                )

                if (
                    victim is not None
                    and attacker is not None
                ):
                    victim_value = abs(
                        PIECE_VALUES[
                            victim.symbol()
                        ]
                    )

                    attacker_value = abs(
                        PIECE_VALUES[
                            attacker.symbol()
                        ]
                    )

                    score += (
                        victim_value
                        - attacker_value // 10
                    )

            if (
                move.promotion
                is not None
            ):
                score += 800

                if (
                    move.promotion
                    == chess.QUEEN
                ):
                    score += 100

            board.push(
                move
            )

            if board.is_checkmate():
                score += 10000

            elif board.is_check():
                score += 500

            board.pop()

            return score

        moves.sort(
            key=move_score,
            reverse=True
        )

        return moves

    def evaluate_terminal_position(
        self,
        fen
    ):
        board = chess.Board(
            fen
        )

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -MATE_SCORE

            return MATE_SCORE

        if board.is_stalemate():
            return 0

        if board.is_insufficient_material():
            return 0

        return None

    def evaluate_material(
        self,
        board
    ):
        score = 0

        for row in board:
            for piece in row:
                if piece is not None:
                    score += (
                        PIECE_VALUES[
                            piece
                        ]
                    )

        return score

    def evaluate_position(
        self,
        board
    ):
        score = 0

        endgame = self.is_endgame(
            board
        )

        for row in range(8):
            for col in range(8):
                piece = board[row][col]

                if piece is None:
                    continue

                piece_type = piece.upper()

                if piece_type == "K":
                    if endgame:
                        table = (
                            KING_ENDGAME_TABLE
                        )
                    else:
                        table = (
                            KING_MIDDLEGAME_TABLE
                        )

                else:
                    table = (
                        PIECE_SQUARE_TABLES.get(
                            piece_type
                        )
                    )

                if table is None:
                    continue

                if piece.isupper():
                    score += (
                        table[row][col]
                    )

                else:
                    mirrored_row = (
                        7 - row
                    )

                    score -= (
                        table[
                            mirrored_row
                        ][
                            col
                        ]
                    )

        return score

    def is_endgame(
        self,
        board
    ):
        non_pawn_material = 0

        for row in board:
            for piece in row:
                if piece is None:
                    continue

                piece_type = (
                    piece.upper()
                )

                if piece_type in {
                    "N",
                    "B",
                    "R",
                    "Q",
                }:
                    non_pawn_material += abs(
                        PIECE_VALUES[
                            piece
                        ]
                    )

        return (
            non_pawn_material
            <= 2600
        )

    def evaluate_mobility(
        self,
        fen
    ):
        board = chess.Board(
            fen
        )

        white_board = (
            board.copy()
        )

        white_board.turn = (
            chess.WHITE
        )

        black_board = (
            board.copy()
        )

        black_board.turn = (
            chess.BLACK
        )

        white_moves = len(
            list(
                white_board.legal_moves
            )
        )

        black_moves = len(
            list(
                black_board.legal_moves
            )
        )

        mobility_difference = (
            white_moves
            - black_moves
        )

        return (
            mobility_difference
            * 5
        )

    def evaluate_pawn_structure(
        self,
        board
    ):
        score = 0

        white_pawns_by_file = [
            []
            for _ in range(8)
        ]

        black_pawns_by_file = [
            []
            for _ in range(8)
        ]

        for row in range(8):
            for col in range(8):
                piece = (
                    board[
                        row
                    ][
                        col
                    ]
                )

                if piece == "P":
                    white_pawns_by_file[
                        col
                    ].append(
                        row
                    )

                elif piece == "p":
                    black_pawns_by_file[
                        col
                    ].append(
                        row
                    )

        for col in range(8):
            white_count = len(
                white_pawns_by_file[
                    col
                ]
            )

            black_count = len(
                black_pawns_by_file[
                    col
                ]
            )

            if white_count > 1:
                score -= (
                    white_count - 1
                ) * 15

            if black_count > 1:
                score += (
                    black_count - 1
                ) * 15

        for col in range(8):
            adjacent_files = []

            if col > 0:
                adjacent_files.append(
                    col - 1
                )

            if col < 7:
                adjacent_files.append(
                    col + 1
                )

            for _ in (
                white_pawns_by_file[
                    col
                ]
            ):
                has_neighbor = any(
                    len(
                        white_pawns_by_file[
                            adjacent_col
                        ]
                    ) > 0
                    for adjacent_col
                    in adjacent_files
                )

                if not has_neighbor:
                    score -= 10

            for _ in (
                black_pawns_by_file[
                    col
                ]
            ):
                has_neighbor = any(
                    len(
                        black_pawns_by_file[
                            adjacent_col
                        ]
                    ) > 0
                    for adjacent_col
                    in adjacent_files
                )

                if not has_neighbor:
                    score += 10

        return score

    def evaluate_king_safety(
        self,
        board
    ):
        score = 0

        white_king = None
        black_king = None

        for row in range(8):
            for col in range(8):
                if (
                    board[
                        row
                    ][
                        col
                    ]
                    == "K"
                ):
                    white_king = (
                        row,
                        col
                    )

                elif (
                    board[
                        row
                    ][
                        col
                    ]
                    == "k"
                ):
                    black_king = (
                        row,
                        col
                    )

        if white_king is not None:
            row, col = (
                white_king
            )

            shield_row = (
                row - 1
            )

            if shield_row >= 0:
                for shield_col in range(
                    max(
                        0,
                        col - 1
                    ),
                    min(
                        8,
                        col + 2
                    )
                ):
                    if (
                        board[
                            shield_row
                        ][
                            shield_col
                        ]
                        == "P"
                    ):
                        score += 10

            if (
                2 <= row <= 5
                and 2 <= col <= 5
            ):
                score -= 20

        if black_king is not None:
            row, col = (
                black_king
            )

            shield_row = (
                row + 1
            )

            if shield_row < 8:
                for shield_col in range(
                    max(
                        0,
                        col - 1
                    ),
                    min(
                        8,
                        col + 2
                    )
                ):
                    if (
                        board[
                            shield_row
                        ][
                            shield_col
                        ]
                        == "p"
                    ):
                        score -= 10

            if (
                2 <= row <= 5
                and 2 <= col <= 5
            ):
                score += 20

        return score

    def evaluate_strategic_features(
        self,
        board
    ):
        score = 0

        white_bishops = 0
        black_bishops = 0

        white_pawns = []
        black_pawns = []

        white_rooks = []
        black_rooks = []

        for row in range(8):
            for col in range(8):
                piece = (
                    board[
                        row
                    ][
                        col
                    ]
                )

                if piece == "B":
                    white_bishops += 1

                elif piece == "b":
                    black_bishops += 1

                elif piece == "P":
                    white_pawns.append(
                        (
                            row,
                            col
                        )
                    )

                elif piece == "p":
                    black_pawns.append(
                        (
                            row,
                            col
                        )
                    )

                elif piece == "R":
                    white_rooks.append(
                        (
                            row,
                            col
                        )
                    )

                elif piece == "r":
                    black_rooks.append(
                        (
                            row,
                            col
                        )
                    )

        if white_bishops >= 2:
            score += 30

        if black_bishops >= 2:
            score -= 30

        for row, col in white_pawns:
            if self.is_passed_pawn(
                row,
                col,
                chess.WHITE,
                black_pawns,
            ):
                advancement = max(
                    0,
                    min(
                        6,
                        6 - row
                    )
                )

                score += (
                    PASSED_PAWN_BONUSES[
                        advancement
                    ]
                )

        for row, col in black_pawns:
            if self.is_passed_pawn(
                row,
                col,
                chess.BLACK,
                white_pawns,
            ):
                advancement = max(
                    0,
                    min(
                        6,
                        row - 1
                    )
                )

                score -= (
                    PASSED_PAWN_BONUSES[
                        advancement
                    ]
                )

        for _, col in white_rooks:
            has_white_pawn = any(
                pawn_col == col
                for _, pawn_col
                in white_pawns
            )

            has_black_pawn = any(
                pawn_col == col
                for _, pawn_col
                in black_pawns
            )

            if (
                not has_white_pawn
                and not has_black_pawn
            ):
                score += 20

            elif not has_white_pawn:
                score += 10

        for _, col in black_rooks:
            has_black_pawn = any(
                pawn_col == col
                for _, pawn_col
                in black_pawns
            )

            has_white_pawn = any(
                pawn_col == col
                for _, pawn_col
                in white_pawns
            )

            if (
                not has_black_pawn
                and not has_white_pawn
            ):
                score -= 20

            elif not has_black_pawn:
                score -= 10

        return score

    def is_passed_pawn(
        self,
        row,
        col,
        color,
        enemy_pawns
    ):
        for (
            enemy_row,
            enemy_col
        ) in enemy_pawns:
            if abs(
                enemy_col - col
            ) > 1:
                continue

            if (
                color
                == chess.WHITE
                and enemy_row < row
            ):
                return False

            if (
                color
                == chess.BLACK
                and enemy_row > row
            ):
                return False

        return True

    def _transposition_key(
        self,
        board
    ):
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            board.ep_square,
            board.halfmove_clock,
        )

    def _reset_search_state(
        self
    ):
        self.transposition_table.clear()
        self.evaluation_cache.clear()

        self.tt_hits = 0
        self.tt_stores = 0
        self.evaluation_cache_hits = 0

    def get_search_stats(
        self
    ):
        return {
            "tt_entries": len(
                self.transposition_table
            ),
            "tt_hits": self.tt_hits,
            "tt_stores": self.tt_stores,
            "evaluation_cache_entries": len(
                self.evaluation_cache
            ),
            "evaluation_cache_hits":
                self.evaluation_cache_hits,
        }

    def _normalize_position(
        self,
        position
    ):
        if isinstance(
            position,
            str
        ):
            return self.fen_to_board(
                position
            )

        if isinstance(
            position,
            list
        ):
            return self._validate_board(
                position
            )

        if isinstance(
            position,
            dict
        ):
            if "fen" in position:
                return self.fen_to_board(
                    position[
                        "fen"
                    ]
                )

            if "board" in position:
                return self._validate_board(
                    position[
                        "board"
                    ]
                )

        raise TypeError(
            "Position must be a FEN string, "
            "8x8 board list, or dict with "
            "'fen'/'board'."
        )

    def _extract_fen(
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
            dict
        ):
            return position.get(
                "fen"
            )

        return None

    def fen_to_board(
        self,
        fen
    ):
        if (
            not fen
            or not isinstance(
                fen,
                str
            )
        ):
            raise ValueError(
                "FEN must be a non-empty string."
            )

        parts = (
            fen.strip().split()
        )

        if not parts:
            raise ValueError(
                "Invalid FEN: empty string."
            )

        board_part = parts[0]

        rows = (
            board_part.split(
                "/"
            )
        )

        if len(rows) != 8:
            raise ValueError(
                "Invalid FEN: board part "
                "must contain 8 rows."
            )

        board = []

        for row in rows:
            parsed_row = []

            for char in row:
                if char.isdigit():
                    parsed_row.extend(
                        [None]
                        * int(char)
                    )

                elif (
                    char
                    in PIECE_VALUES
                ):
                    parsed_row.append(
                        char
                    )

                else:
                    raise ValueError(
                        f"Invalid FEN: "
                        f"unknown piece symbol "
                        f"'{char}'."
                    )

            if (
                len(parsed_row)
                != 8
            ):
                raise ValueError(
                    "Invalid FEN: each row "
                    "must contain exactly "
                    "8 squares."
                )

            board.append(
                parsed_row
            )

        return board

    def _validate_board(
        self,
        board
    ):
        if len(board) != 8:
            raise ValueError(
                "Board must have 8 rows."
            )

        for row in board:
            if len(row) != 8:
                raise ValueError(
                    "Each board row must "
                    "have 8 columns."
                )

            for square in row:
                if (
                    square is not None
                    and square
                    not in PIECE_VALUES
                ):
                    raise ValueError(
                        f"Invalid board "
                        f"piece symbol: "
                        f"{square}"
                    )

        return board