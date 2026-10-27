# Chess Vision Project

System for automatic chess position recognition from images and position analysis.

## Current pipeline

Image
→ chessboard detection
→ perspective correction
→ YOLO-based chess piece detection
→ mapping pieces to board squares
→ board representation
→ FEN generation
→ position validation
→ custom evaluator / Stockfish

## Project structure

- `src/chess_logic` - board representation, FEN and custom evaluation
- `src/chess_engine` - Stockfish integration
- `src/vision` - chessboard localization and perspective correction
- `src/piece_detection` - chess piece detection and board mapping
- `scripts` - dataset preparation, analysis and model training
- `tests` - unit and integration tests

## Tests

Run all tests:

```bash
pytest -v
