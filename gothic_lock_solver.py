import argparse
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path


WIDTH = 7

# In the game the visible movement is reversed compared with the solver model.
MODEL_LEFT_KEY = "d"
MODEL_RIGHT_KEY = "a"
NEXT_ROW_KEY = "w"
PREVIOUS_ROW_KEY = "s"


@dataclass(frozen=True)
class Move:
    row: int
    direction: int


def load_puzzle(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    matrix = data["matrix"]
    if not matrix:
        raise ValueError("matrix must contain at least one row")
    if any(len(row) != WIDTH for row in matrix):
        raise ValueError(f"each matrix row must contain exactly {WIDTH} values")

    positions = []
    for row_index, row in enumerate(matrix):
        ones = [column for column, value in enumerate(row) if value == 1]
        if len(ones) != 1:
            raise ValueError(
                f"row {row_index + 1} must contain exactly one 1, found {len(ones)}"
            )
        if any(value not in (0, 1) for value in row):
            raise ValueError("matrix can contain only 0 and 1")
        positions.append(ones[0])

    target_column = int(data.get("target_column", 4))
    if not 1 <= target_column <= WIDTH:
        raise ValueError(f"target_column must be from 1 to {WIDTH}")

    row_count = len(matrix)
    effects = data.get("effects")
    if effects is None:
        effects = [[0 for _ in range(row_count)] for _ in range(row_count)]
    validate_effects(effects, row_count)

    return tuple(positions), target_column - 1, tuple(tuple(row) for row in effects)


def validate_effects(effects, row_count: int):
    if len(effects) != row_count:
        raise ValueError("effects must contain one row per matrix row")

    for row_index, row in enumerate(effects):
        if len(row) != row_count:
            raise ValueError(
                f"effects row {row_index + 1} must contain {row_count} values"
            )
        for value in row:
            if value not in (-1, 0, 1):
                raise ValueError("effects values must be -1, 0, or 1")


def apply_move(state, effects, move: Move):
    next_state = list(state)
    row_effects = effects[move.row]
    next_state[move.row] += move.direction

    for row_index, effect in enumerate(row_effects):
        if row_index == move.row:
            continue
        next_state[row_index] += move.direction * effect

    if any(position < 0 or position >= WIDTH for position in next_state):
        return None
    return tuple(next_state)


def solve(initial_state, target_column: int, effects):
    goal = tuple(target_column for _ in initial_state)
    if initial_state == goal:
        return []

    start = (initial_state, None)
    queue = [(0, 0, 0, initial_state, None)]
    best = {start: (0, 0)}
    previous = {start: None}
    sequence = 1
    row_count = len(initial_state)
    best_goal = None

    while queue:
        move_count, group_count, _, state, last_move = heapq.heappop(queue)
        search_key = (state, last_move)
        if best.get(search_key) != (move_count, group_count):
            continue

        if state == goal:
            best_goal = search_key
            break

        for row in range(row_count):
            for direction in (-1, 1):
                move = Move(row=row, direction=direction)
                next_state = apply_move(state, effects, move)
                if next_state is None:
                    continue

                next_key = (next_state, move)
                next_cost = (
                    move_count + 1,
                    group_count if move == last_move else group_count + 1,
                )
                if next_cost >= best.get(next_key, (10**9, 10**9)):
                    continue

                best[next_key] = next_cost
                previous[next_key] = (search_key, move)
                heapq.heappush(
                    queue,
                    (next_cost[0], next_cost[1], sequence, next_state, move),
                )
                sequence += 1

    return restore_path(previous, best_goal) if best_goal else None


def restore_path(previous, search_key):
    moves = []
    while True:
        prior = previous[search_key]
        if prior is None:
            return list(reversed(moves))
        prior_key, move = prior
        moves.append(move)
        search_key = prior_key


def compact_moves(moves):
    lines = []
    index = 0

    while index < len(moves):
        move = moves[index]
        count = 1
        while index + count < len(moves) and moves[index + count] == move:
            count += 1

        direction = "right" if move.direction > 0 else "left"
        suffix = f" x{count}" if count > 1 else ""
        lines.append(f"- Row {move.row + 1}: {direction}{suffix}")
        index += count

    return lines


def build_key_presses(moves, start_row=0):
    keys = []
    selected_row = start_row

    for move in moves:
        while selected_row < move.row:
            keys.append(NEXT_ROW_KEY)
            selected_row += 1
        while selected_row > move.row:
            keys.append(PREVIOUS_ROW_KEY)
            selected_row -= 1

        keys.append(MODEL_LEFT_KEY if move.direction < 0 else MODEL_RIGHT_KEY)

    return keys


def compact_keys(keys):
    lines = []
    index = 0

    while index < len(keys):
        key = keys[index]
        count = 1
        while index + count < len(keys) and keys[index + count] == key:
            count += 1

        suffix = f" x{count}" if count > 1 else ""
        lines.append(f"- {key.upper()}{suffix}")
        index += count

    return lines


def print_solution(moves):
    if moves is None:
        print("No solution found.")
        return

    if not moves:
        print("Already solved.")
        return

    print(f"Moves ({len(moves)}):")
    for line in compact_moves(moves):
        print(line)

    key_presses = build_key_presses(moves)
    print()
    print(f"Keys ({len(key_presses)} presses, starting from row 1):")
    for line in compact_keys(key_presses):
        print(line)


def press_keys(keys, delay, start_delay):
    try:
        import pydirectinput
    except ImportError:
        raise SystemExit(
            "Module pydirectinput is not installed. Install it with: "
            "pip install pydirectinput"
        )

    print(f"Starting in {start_delay:g} seconds. Focus the game window now.")
    time.sleep(start_delay)

    for key in keys:
        pydirectinput.press(key)
        time.sleep(delay)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Solve and optionally execute a Gothic-style lock puzzle."
    )
    parser.add_argument("puzzle", type=Path, help="Path to puzzle JSON file.")
    parser.add_argument(
        "--press",
        action="store_true",
        help="Actually press the generated keys. Focus the game window first.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.08,
        help="Delay between key presses when --press is used.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Delay before pressing starts when --press is used.",
    )
    args = parser.parse_args()

    try:
        initial_state, target_column, effects = load_puzzle(args.puzzle)
        moves = solve(initial_state, target_column, effects)
        print_solution(moves)
        if args.press and moves:
            press_keys(build_key_presses(moves), args.delay, args.start_delay)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()
