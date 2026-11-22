from pathlib import Path
import csv
import random

import chess


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_SET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "engine_positions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "engine_test_positions.csv"
)

SEED = 20261122

TARGETS = {
    "opening": 50,
    "middlegame": 60,
    "endgame": 40,
}

MAX_GAMES = 20000
MAX_PLIES_PER_GAME = 200

MIN_SAVED_PLY = 6


def normalize_fen(fen):
    parts = fen.split()

    return " ".join(
        parts[:4]
    )


def load_existing_positions():
    positions = set()

    if not DEV_SET_PATH.exists():
        return positions

    with open(
        DEV_SET_PATH,
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            fen = row.get("fen")

            if fen:
                positions.add(
                    normalize_fen(
                        fen
                    )
                )

    return positions


def get_non_pawn_material(board):
    values = {
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }

    total = 0

    for piece_type, value in values.items():
        white_count = len(
            board.pieces(
                piece_type,
                chess.WHITE
            )
        )

        black_count = len(
            board.pieces(
                piece_type,
                chess.BLACK
            )
        )

        total += (
            white_count
            + black_count
        ) * value

    return total


def classify_phase(board):
    ply = board.ply()

    non_pawn_material = (
        get_non_pawn_material(
            board
        )
    )

    queens = (
        len(
            board.pieces(
                chess.QUEEN,
                chess.WHITE
            )
        )
        + len(
            board.pieces(
                chess.QUEEN,
                chess.BLACK
            )
        )
    )

    total_pieces = len(
        board.piece_map()
    )

    if (
        ply <= 20
        and non_pawn_material >= 5000
    ):
        return "opening"

    if (
        non_pawn_material <= 2600
        or total_pieces <= 12
        or (
            queens == 0
            and non_pawn_material <= 3600
        )
    ):
        return "endgame"

    if ply > 12:
        return "middlegame"

    return None


def choose_random_move(board):
    moves = list(
        board.legal_moves
    )

    if not moves:
        return None

    weights = []

    for move in moves:
        weight = 1.0

        if board.is_capture(move):
            weight += 2.0

            victim = board.piece_at(
                move.to_square
            )

            if victim is not None:
                if victim.piece_type == chess.QUEEN:
                    weight += 2.0

                elif victim.piece_type == chess.ROOK:
                    weight += 1.0

        if move.promotion is not None:
            weight += 3.0

        board.push(move)

        if board.is_checkmate():
            weight += 4.0

        elif board.is_check():
            weight += 0.5

        board.pop()

        weights.append(
            weight
        )

    return random.choices(
        moves,
        weights=weights,
        k=1
    )[0]


def targets_reached(counts):
    return all(
        counts[phase]
        >= TARGETS[phase]
        for phase in TARGETS
    )


def phase_needed(
    phase,
    counts
):
    return (
        counts[phase]
        < TARGETS[phase]
    )


def collect_game_candidates(
    game_number,
    dev_positions,
    generated_positions
):
    board = chess.Board()

    candidates = {
        "opening": [],
        "middlegame": [],
        "endgame": [],
    }

    for _ in range(
        MAX_PLIES_PER_GAME
    ):
        if board.is_game_over():
            break

        if board.ply() >= MIN_SAVED_PLY:
            phase = classify_phase(
                board
            )

            if phase is not None:
                fen = board.fen()

                normalized = (
                    normalize_fen(
                        fen
                    )
                )

                if (
                    board.is_valid()
                    and normalized
                    not in dev_positions
                    and normalized
                    not in generated_positions
                ):
                    candidates[
                        phase
                    ].append(
                        {
                            "fen": fen,
                            "phase": phase,
                            "game": game_number,
                            "ply": board.ply(),
                        }
                    )

        move = choose_random_move(
            board
        )

        if move is None:
            break

        board.push(
            move
        )

    return candidates


def choose_candidate(
    candidates,
    counts
):
    available_phases = [
        phase
        for phase in TARGETS
        if (
            phase_needed(
                phase,
                counts
            )
            and candidates[
                phase
            ]
        )
    ]

    if not available_phases:
        return None

    def remaining_ratio(phase):
        remaining = (
            TARGETS[phase]
            - counts[phase]
        )

        return (
            remaining
            / TARGETS[phase]
        )

    max_ratio = max(
        remaining_ratio(
            phase
        )
        for phase in available_phases
    )

    priority_phases = [
        phase
        for phase in available_phases
        if remaining_ratio(
            phase
        ) == max_ratio
    ]

    selected_phase = random.choice(
        priority_phases
    )

    return random.choice(
        candidates[
            selected_phase
        ]
    )


def generate_positions():
    random.seed(
        SEED
    )

    dev_positions = (
        load_existing_positions()
    )

    generated_positions = set()

    results = []

    counts = {
        "opening": 0,
        "middlegame": 0,
        "endgame": 0,
    }

    game_number = 0

    while (
        not targets_reached(
            counts
        )
        and game_number < MAX_GAMES
    ):
        game_number += 1

        candidates = (
            collect_game_candidates(
                game_number,
                dev_positions,
                generated_positions
            )
        )

        selected = choose_candidate(
            candidates,
            counts
        )

        if selected is None:
            continue

        normalized = normalize_fen(
            selected["fen"]
        )

        generated_positions.add(
            normalized
        )

        phase = selected[
            "phase"
        ]

        counts[
            phase
        ] += 1

        results.append(
            selected
        )

        print(
            f"Game {game_number:5} | "
            f"{phase:10} "
            f"{counts[phase]:3}/"
            f"{TARGETS[phase]} | "
            f"ply {selected['ply']:3} | "
            f"{selected['fen']}"
        )

    if not targets_reached(
        counts
    ):
        raise RuntimeError(
            "Could not generate enough "
            "independent positions. "
            "Increase MAX_GAMES."
        )

    random.shuffle(
        results
    )

    return (
        results,
        counts,
        game_number
    )


def save_positions(
    positions
):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "phase",
                "game",
                "ply",
                "fen",
            ]
        )

        writer.writeheader()

        for index, row in enumerate(
            positions,
            start=1
        ):
            writer.writerow(
                {
                    "id": index,
                    "phase": row[
                        "phase"
                    ],
                    "game": row[
                        "game"
                    ],
                    "ply": row[
                        "ply"
                    ],
                    "fen": row[
                        "fen"
                    ],
                }
            )


def main():
    (
        positions,
        counts,
        games_generated
    ) = generate_positions()

    save_positions(
        positions
    )

    unique_games = len(
        {
            position["game"]
            for position
            in positions
        }
    )

    print()
    print("=" * 60)
    print(
        "INDEPENDENT TEST SET GENERATED"
    )
    print("=" * 60)

    print(
        f"Opening:    "
        f"{counts['opening']}"
    )

    print(
        f"Middlegame: "
        f"{counts['middlegame']}"
    )

    print(
        f"Endgame:    "
        f"{counts['endgame']}"
    )

    print(
        f"Total:      "
        f"{len(positions)}"
    )

    print(
        f"Unique games: "
        f"{unique_games}"
    )

    print(
        f"Games simulated: "
        f"{games_generated}"
    )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()