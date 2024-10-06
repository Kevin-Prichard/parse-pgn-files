#!/usr/bin/env python3


QUEEN_DELTAS = (
    (-1, 1), (0, 1), (1, 1),
    (-1, 0),           (1, 0),
    (-1, -1), (0, -1), (1, -1))
QD = QUEEN_DELTAS


def whats_what(opking, start, dx, dy):
    kdx, kdy = opking[0] - start[0], opking[1] - start[1]

    # could this be a pin line?
    return kdx, kdy, (
            ((kdx == 0 and dx == 0) or (kdx and (kdx / abs(kdx))) == dx) and
            ((kdy == 0 and dy == 0) or (kdy and (kdy / abs(kdy))) == dy))


def exercise_exercise_everybody_exercise():
    king_at = 4, 4

    for qdx, qdy in QUEEN_DELTAS:
        for depth in range(1, 3):
            queen_at = king_at[0] + qdx * depth, king_at[1] + qdy * depth
            for dx, dy in QUEEN_DELTAS:
                print(f"depth: {depth}, delta: {qdx}, {qdy}, "
                      f"queen: {queen_at}, kdelta: ",
                      whats_what(king_at, queen_at, dx, dy))


if __name__ == '__main__':
    exercise_exercise_everybody_exercise()
