#!/usr/bin/env python3
#cython: language_level=3

import argparse
import copy
from collections import defaultdict as dd
from enum import Enum
import json
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import re
import sys
import time
from typing import List, Tuple, Callable, Text, Dict

from pgn_parser import pgn_file
from move_parser import agn, FILE_CHARS
from board_demo import prepare_board

from colored import fore, back, style
from parsita import Success

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


NAUGHTY_CHARS = re.compile(b'[^\n -~]+')


# PGN file sections can be identified by the first character of the line,
# which lets us break a PGN file into its constituent parts

class PGNParts(Enum):
    ANNOTATION = 1
    MOVES = 2
    NEWLINE = 3


LINE_TYPE = {
    b'': PGNParts.ANNOTATION,  # Empty line w/o \n signals end-of-file & treated same as annotation
    b'[': PGNParts.ANNOTATION,
    b'1': PGNParts.MOVES,
    b'\n': PGNParts.NEWLINE,
}


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()


def get_args(argv: List[str]) -> Tuple[argparse.Namespace, ArgumentParser]:
    global args
    parser = ArgumentParser(
        prog='./parse_pgn.py',
        description='Parse PGN file(s) and stream(s): '
                    'parse PGNs and pass the resulting '
                    'parse tree to a function.')
    parser.add_argument('--file', '-f', dest='pgn_path',
                        type=str, action='store', default='-',
                        help="File from which one or more PGN blocks will be "
                             "read.  For POSIX pipes use -f /dev/stdin")
    parser.add_argument('--output', '-o', dest='out_path',
                        type=str, action='store', default='/dev/stdout',
                        help="File to which JSON output will be written.")
    parser.add_argument('--processes', '-p', dest='process_count',
                        type=int, action='store', default=mp.cpu_count(),
                        help='Number of processes to use; 1=single tasking')
    parser.add_argument('--limit', '-l', dest='pgn_limit',
                        type=int, action='store', default=0,
                        help="Stop reading pgn_path after this many PGNs.")
    parser.add_argument('--queue_limit', '-q', dest='queue_limit',
                        type=int, action='store', default=250,
                        help="Pause enqueing when all queues are this full")
    parser.add_argument('--show', '-s', dest='show_board',
                        action='store_true', default=False,
                        help="Display board after each side moves")
    parser.add_argument('--inter', '-i', dest='interactive',
                        action='store_true', default=False,
                        help="Wait for user to hit enter after each move")
    parser.add_argument('--colors', '-c', dest='colors',
                        type=str, action='store', default='blue_purple',
                        help="Pick a color scheme for ANSI console: "
                             "black_white, green_gold, blue_purple")
    parser.add_argument('--style', '-y', dest='piece_style',
                        type=str, action='store', default='solid',
                        help="Choose Unicode piece style: solid or outline")
    parser.add_argument('--debug', '-d', dest='game_move_debug',
                        type=str, action='store', default=None,
                        help="Process only this game number's PGN starting at "
                             "move #, i.e. -d 3,19 debugs at game 3, move 19 ")

    args = parser.parse_args(argv)
    return args, parser


COLOR_SCHEMES = {
    # black & white: piece colors
    # dark & light: space or square colors
    'black_white': (black_white:={
        'black': 'black',
        'white': 'white',
        'dark': 'black',
        'light': 'white',
    }),
    'bw': black_white,
    'green_gold': {
        'black': 'dark_green',
        'white': 'gold_1',
        'dark': 'dark_olive_green_1b',
        'light': 'tan',
    },
    'blue_purple': {
        'black': 'purple_1a',
        'white': 'gold_3a',
        'dark': 'dark_blue',
        'light': 'sky_blue_1',
    },
    # '': {
    #     'black': '',
    #     'white': '',
    #     'dark_space': '',
    #     'light_space': '',
    # },
}

SQUARE_COL = {0: 'light', 1: 'dark'}
PIECE_COL = {'W': 'white', 'B': 'black'}


class PGNStreamSlicer:
    def __init__(self, pgn_path: str, parse_cb: Callable[[Dict], None] = None):
        self.pgn_path = pgn_path

        # "rb": PGNs are UTF-8 encoded, e.g. "Réti Opening"
        self.pgn_file = open(self.pgn_path, 'rb')

        self.parse_cb = parse_cb
        self.result = None

    def next(self):
        # Saw off PGN hunks and hand them back to caller
        buf = []
        this_type, prev_type = None, None
        keep_going = True
        while keep_going:
            line = self.pgn_file.readline()
            this_type = LINE_TYPE.get(line[:1])
            if not line:
                keep_going = False
            if (this_type == PGNParts.ANNOTATION and
                    prev_type == PGNParts.MOVES ):
                final = b"\n".join(buf)
                # When annotations contain non-ASCII characters, wipe them out
                # because parsita doesn't handle them well.
                final_utf8 = NAUGHTY_CHARS.sub(
                    b'_', final, 999).decode('utf-8')
                yield final_utf8
                buf.clear()
                buf = [line[:-1]]
            else:
                buf.append(line[:-1])

            if this_type is not PGNParts.NEWLINE:
                prev_type = this_type


start_time = last_time = time.time()
todo = 34869171

m = mp.Manager()

POINTS = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}

"""
PROBLEM: we don't know which piece is is being taken, some of the time.
We need a basic headless board to know what piece exists at a location,
and then we can determine the value of the move.
"""


class Piece:
    def __init__(self, piece: str, side: str, rank: int, file: int):
        self.side = side  # 'B' or 'W'
        self.piece = piece
        self.rank = rank
        self.file = file

    def move_pattern(self):
        pass

    def move(self, square):
        self.file = square[0]
        self.rank = int(square[1])

    def __repr__(self):
        return f"{self.piece} at {self.rank}, {self.file}"


FILES = {f: i for i, f in enumerate(FILE_CHARS)}
BOARD_STD = [
    ['RW', 'NW', 'BW', 'QW', 'KW', 'BW', 'NW', 'RW'],
    ['PW'] * 8,
    EMPTY := [None] * 8,
    copy.copy(EMPTY),
    copy.copy(EMPTY),
    copy.copy(EMPTY),
    ['PB'] * 8,
    ['RB', 'NB', 'BB', 'QB', 'KB', 'BB', 'NB', 'RB'],
]


FILE_BASE = ord('a')
OPPO = {'W': 'B', 'B': 'W'}
SIDE_REPR = {'W': 'white', 'B': 'black', 'white': 'W', 'black': 'B'}
DISPLAY = {
    'solid': {'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',},
    'outline': {'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',}
}
PSG = {
    "PB": 1, "NB": 2, "BB": 3, "RB": 4, "QB": 5, "KB": 6,
    "PW": 7, "NW": 8, "BW": 9, "RW": 10, "QW": 11, "KW": 12,
    None: 0, "": 0,
}
en_passant = 0


def mk_coords(square):
    # returns array coordinates for self.board, (0..7, 0..7)
    return ord(square[0])-FILE_BASE, int(square[1])-1


def mk_square(*coords):
    # returns algebraic: [a..h][1..8] from board coords (0..7, 0..7)
    assert 0 <= coords[0] <= 7 and 0 <= coords[1] <= 7, f"mk_square out-of-bounds: {coords}"
    return f"{chr(FILE_BASE+coords[0])}{coords[1]+1}"


# piece movement destinations based on a set of x,y coordination deltas
def dests_common(square, coord_alters):
    dests = set()
    start = mk_coords(square)
    for x, y in coord_alters:
        new_file, new_rank = start[0]+x, start[1]+y
        if 0 <= new_file <=7 and 0 <= new_rank <=7:
            dests.add(mk_square(new_file, new_rank))

    return dests


def dests_radial(piece, square, coord_alters, action, board, side):
    # piece movement destinations based on radial delta paths
    # e.g. queen moves along eight radial paths relative to current square
    # Important: the in-memory board's origin is a1 aka board.board[0][0]
    # which is the lower-left corner. coord_alters pairs are relative to this.
    dests = set()
    start = mk_coords(square)
    oking = mk_coords(board.by_piece('K', OPPO[side])[0])
    # if debug_this.now():
    #     import pudb; pu.db
    kx, ky = start[0] - oking[0], start[1] - oking[1]
    # x,y are the change at each step
    once = False
    for dx, dy in coord_alters:
        # if (max(start[0], oking[0]) - min(start[0], oking[0]))
        is_pin_line = (kx and kx/abs(kx) == dx
                       and ky and ky/abs(ky) == dy and abs(kx) == abs(ky))
        if debug_this.now():
            if not once:
                print(board)
                print(square, side, action)
                once = True
            print(is_pin_line, kx, ky, dx, dy,
                  kx/dx if dx else None,
                  ky/dy if dy else None,
                  kx/abs(kx) == dx if kx else None,
                  ky/abs(ky) == dy if ky else None, "*"*50)
        keep_on = True
        # starting point is current piece square
        new_file, new_rank = start[0], start[1]
        # keep stepping along the radial until out-of-bounds
        # we are not checking for piece collisions, as these are PGN playbacks
        while keep_on:
            new_file += dx
            new_rank += dy
            if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                new_target = board.get_coords(file=new_file, rank=new_rank)
                if action == 'move':
                    if new_target is None and not board.is_pinned(piece, square, side):
                        dests.add(mk_square(new_file, new_rank))
                    elif is_pin_line:
                        board.set_pin(
                            pinned_piece=new_target[0], side=OPPO[side],
                            pinned_sq=mk_square(new_file, new_rank),
                            pinning_piece=board.get(square)[0],
                            pinning_sq=square,
                            king_sq=mk_square(*oking)
                        )
                elif action == 'capture':
                    if new_target:
                        if new_target[1] == OPPO[side]:
                            dests.add(mk_square(new_file, new_rank))
                        else:
                            keep_on = False
                else:
                    keep_on = False
            else:
                keep_on = False

    if debug_this.now():
        import pudb; pu.db
    return dests


def dests_knight(square):
    return dests_common(square,
                        ((-2, 1), (-1, 2), (1, 2), (2, 1),
                         (-2, -1), (-1, -2), (1, -2), (2, -1)))


def dests_king(square):
    return dests_common(square,
                        ((-1, 1,), (0, 1), (1, 1),
                         (-1, 0), (1, 0),
                         (-1, -1), (0, -1), (1, -1)))


def dests_bishop(square, action, board, side):
    # Clockwise from 10:30
    # if debug_this.now():
    #     import pudb; pu.db
    return dests_radial("B", square,
                        ((-1, 1), (1, 1),
                         (1, -1), (-1, -1)),
                        action=action, board=board, side=side)


def dests_rook(square, action, board, side):
    # Clockwise from 9:00
    return dests_radial("R", square,
                        ((-1, 0), (0, 1),
                         (1, 0), (0, -1)),
                        action=action, board=board, side=side)


def dests_queen(square, action, board, side):
    # clockwise from 10:30
    return dests_radial("Q", square,
                        ((-1, 1), (0, 1), (1, 1),
                         (-1, 0),         (1, 0),
                         (-1, -1), (0, -1), (1, -1)),
                        action=action, board=board, side=side)


def dests_pawn(square, side, action, board=None, target=None):
    dests = set()
    start = mk_coords(square)
    capture_dir = None
    if side == 'W':
        if action == 'move':
            # move
            if start[1] == 1:
                if not board.get(tsq := mk_square(start[0], 2)):
                    dests.add(tsq)
                    if not board.get(tsq := mk_square(start[0], 3)):
                        dests.add(tsq)
            else:
                if not board.get(tsq := mk_square(start[0], start[1] + 1)):
                    dests.add(tsq)
        elif action == 'capture':
            # capture
            capture_dir = 1
    elif side == 'B':
        if action == 'move':
            # move
            if start[1] == 6:
                if not board.get(tsq := mk_square(start[0], 5)):
                    dests.add(tsq)
                    if not board.get(tsq := mk_square(start[0], 4)):
                        dests.add(tsq)
            else:
                if not board.get(tsq := mk_square(start[0], start[1] - 1)):
                    dests.add(tsq)
        elif action == 'capture':
            # capture
            capture_dir = -1
    else:
        raise ValueError(f"Unexpected side: {side}")

    if action == 'capture':
        if start[0] > 0:
            if board.get(tsq := mk_square(start[0] - 1, start[1] + capture_dir)):
                dests.add(tsq)
        if start[0] < 7:
            if board.get(tsq := mk_square(start[0] + 1, start[1] + capture_dir)):
                dests.add(tsq)

    return dests


class Board:
    def __init__(self):
        self.board = copy.deepcopy(BOARD_STD)
        self._pieces = dd(set)
        self._populate_pieces()
        self._pinned = dict()
        self._pinner = dict()
        # self._gui_board, self._gui_board_window = prepare_board()

    def _populate_pieces(self):
        for rank in (0, 1, 6, 7):
            for file in range(8):
                piece = self.board[rank][file]
                if piece:
                    self._pieces[piece].add(mk_square(file, rank))

    def reposition(self, from_sq, to_sq, is_capture=False):
        try:
            from_piece = self.get(from_sq)
            to_piece = self.get(to_sq)
            if to_piece:
                if is_capture:
                    self._pieces[to_piece].remove(to_sq)
                else:
                    raise ValueError(
                        f"Destination square occupied on move: "
                        f"{to_piece}{to_sq}")
            self._pieces[from_piece].remove(from_sq)
            self._pieces[from_piece].add(to_sq)
            old_coords = mk_coords(from_sq)
            new_coords = mk_coords(to_sq)
            self.board[old_coords[1]][old_coords[0]] = None
            self.board[new_coords[1]][new_coords[0]] = from_piece
            if self.is_pinner(from_piece[0], from_piece[1], from_sq):
                self.del_pin(from_piece[0], from_sq, from_piece[1])

        except Exception as ee:
            import pudb; pu.db
            x = 1

    def redraw_gui_board(self):
        for rank in range(8):
            for file in range(8):
                self._gui_board.psg_board[rank][file] = PSG[self.get_coords(rank, file)]
        self._gui_board.redraw_board(self._gui_board_window)

    def get(self, square):
        return self.board[int(square[1])-1][ord(square[0])-FILE_BASE]

    def get_coords(self, rank, file):
        return self.board[rank][file]

    def set(self, square, piece=None, side=None):
        try:
            self.board[int(square[1])-1][ord(square[0])-FILE_BASE] = f"{piece}{side}" if piece else None
        except Exception as ee:
            print(self)
            import pudb; pu.db
            x = 1

    def set_pin(self, pinned_piece, side, pinned_sq, pinning_piece, pinning_sq, king_sq):
        self._pinned[f"{pinned_piece}{side}"] = {
            'pinned_sq': pinned_sq,
            'pinning_piece': pinning_piece,
            'pinning_sq': pinning_sq,
            'king_sq': king_sq,
            'side': side,
        }
        self._pinner[f"{pinning_piece}{side}"] = {
            'pinned_sq': pinned_sq,
            'pinning_piece': pinning_piece,
            'pinning_sq': pinning_sq,
            'king_sq': king_sq,
            'side': side,
        }

    def unset_pin(self, pinned_piece, square, side):
        pin = f"{pinned_piece}{side}"
        if pin in self._pinned and self._pinned[pin]['pinned_sq'] == square:
            del self._pinned[pin]

    def is_pinned(self, piece, side, square):
        pin = f"{piece}{side}"
        return pin in self._pinned and self._pinned[pin]['pinned_sq'] == square

    def is_pinner(self, piece, side, square):
        pin = f"{piece}{side}"
        return pin in self._pinner and self._pinner[pin]['pinning_sq'] == square

    def by_file(self, file, piece, side):
        ps = f"{piece}{side}"
        res = []
        file_no = ord(file) - FILE_BASE
        for rank in range(8):
            if self.board[rank][file_no] == ps:
                res.append(f"{file}{rank+1}")
        return res

    def by_rank(self, rank, piece, side):
        ps = f"{piece}{side}"
        res = []
        rank_no = int(rank) - 1
        for file in FILE_CHARS:
            if self.board[rank_no][ord(file) - FILE_BASE] == ps:
                res.append(f"{file}{rank_no+1}")
        return res

    def by_piece(self, piece, side):
        return sorted(self._pieces[f"{piece}{side}"])

    def by_piece_(self, piece, side):
        ps = f"{piece}{side}"
        res = []
        for rank in range(8):
            for file in range(8):
                if self.board[rank][file] == ps:
                    res.append(f"{chr(FILE_BASE+file)}{rank+1}")
        return res

    def piece_dests(self, piece, start_square,
                    action=None, side=None, board=None, target=None):

        match piece[0]:
            case 'N':
                return dests_knight(start_square)
            case 'B':
                return dests_bishop(start_square, action=action, board=board, side=side)
            case 'R':
                return dests_rook(start_square, action=action, board=board, side=side)
            case 'P':
                return dests_pawn(start_square, action=action, side=side, target=target, board=board)
            case 'Q':
                return dests_queen(start_square, action=action, board=board, side=side)
            case 'K':
                return dests_king(start_square)
        raise ValueError(f"Unexpected piece type: {piece}, {side} {start_square}")

    def castle(self, m, side):
        rank = 1 if side == 'W' else 8
        king_origin = f'e{rank}'
        # kingside
        if m['target'] == 'O-O':
            king_dest = f'g{rank}'
            rook_origin = f'h{rank}'
            rook_dest = f'f{rank}'
        # queenside
        elif m['target'] == 'O-O-O':
            king_dest = f'c{rank}'
            rook_origin = f'a{rank}'
            rook_dest = f'd{rank}'
        else:
            raise ValueError(f"Attempted castling but no joy: {m}")
        self.reposition(king_origin, king_dest)
        self.reposition(rook_origin, rook_dest)
        # self.set(king_origin, None)
        # self.set(rook_origin, None)
        # self.set(king_dest, 'K', side=side)
        # self.set(rook_dest, 'R', side=side)

    def move(self, m, side):
        global en_passant
        points = 0
        old, new = None, None

        if debug_this.now():
            import pudb; pu.db

        if (m['action'] == 'capture' and m['piece'] == 'P'
                and not self.get(m['target'])):
            # en passant always comes from, and targets,
            # the 4th (W) or 5th (B) rank
            rank = '4' if side == 'B' else '5'
            old = m['actor_file'] + rank
            new = m['target']
            self.set(m['target'][0] + rank, None)
            self._pieces[f"P{OPPO[side]}"].remove(m['target'][0] + rank)
            en_passant += 1
            points = 1

        elif m['action'] == 'promote' and 'target' in m:
            dests = self.by_file(m['target'][0], m['piece'], side)
            # import pudb; pu.db
            self.set(dests[0], piece=None)
            self.set(m['target'], piece=m['replacement'], side=side)
            self._pieces[f"{m['piece']}{side}"].remove(dests[0])
            self._pieces[f"{m['replacement']}{side}"].add(m['target'])

        elif 'actor_file' in m:
            squares = self.by_file(m['actor_file'], m['piece'], side)

            if m['action'] == 'promote':
                if len(squares) != 1:
                    import pudb; pu.db

            # Iterate current origin squares of matching pieces
            for piece_square in squares:

                # Is the current move's destination in one of the piece's
                # possible destinations?
                dests = self.piece_dests(
                        piece=m['piece'], start_square=piece_square,
                        side=side, action=m['action'], board=self)

                if m['action'] == 'promote':
                    import pudb; pu.db
                    old = piece_square
                    new = dests[0]
                    break
                elif m['target'] in dests:
                    old = piece_square
                    new = m['target']
                    break

        elif 'actor_rank' in m:
            squares = self.by_rank(m['actor_rank'], m['piece'], side)
            # Iterate current origin squares of matching pieces
            for piece_square in squares:

                # Is the current move's destination in one of the piece's
                # possible destinations?
                if m['target'] in self.piece_dests(
                        piece=m['piece'], start_square=piece_square,
                        side=side, action=m['action'], board=self):
                    old = piece_square
                    new = m['target']
                    break

        elif 'actor_square' in m:
            old = m['actor_square']
            new = m['target']

        elif m['piece'] == 'K' and m['target'][0] == 'O':
            self.castle(m, side=side)
            return 0

        else:
            # Locate the piece: we know its type, but its origin square,
            # rank or file wasn't mentioned in the algebraic.
            # This block results from matching the rule:
            # (piece & square > move_singular)

            # Find squares of the piece type
            squares = self.by_piece(m['piece'], side)

            # This guardrail should only indicate a bug in my code
            if not squares:
                raise ValueError(f"No pieces of type {m} on board")

            # Iterate current origin squares of matching pieces
            for piece_square in squares:

                # Is the current move's destination in one of the piece's
                # possible destinations?
                if (m['piece'] == 'P' and 'target' in m
                        and piece_square[0] != m['target'][0]):
                    continue
                dests = self.piece_dests(
                    piece=m['piece'], start_square=piece_square,
                    side=side, action=m['action'], board=self,
                    target=m['target'])
                if m['target'] in dests and not self.is_pinned(m['piece'], side, piece_square):
                    old = piece_square
                    new = m['target']
                    break

            if old is None or new is None:
                import pudb; pu.db
                raise ValueError(f"Moving piece isn't found: {m}")

        if m['action'] == 'capture':
            cap_target = self.get(m['target'])
            points += POINTS[cap_target[0]] if cap_target else 0

        if m['action'] != 'promote':
            self.reposition(old, new, is_capture=m['action'] == 'capture')
            # self.set(old, None)
            # self.set(new, m['piece'], side)

        return points

    def __repr__(self):
        res = []
        cols = COLOR_SCHEMES[args.colors]
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq_col = int(not((rank + file) % 2))
                piece = self.board[rank][file]
                row.append(
                    f"{back(cols[SQUARE_COL[sq_col]])}"
                    f"{fore(cols[PIECE_COL[piece[1] if piece else 'B']])}"
                    f"{DISPLAY[args.piece_style][piece[0]] if piece else ' ':1s}")
                sq_col = int(not(sq_col))
            row.append(f"{back('black')}{fore('white')}")
            res.append(" ".join(row))
        return "\n".join(res)


def board(moves, graph):
    global en_passant
    b = Board()
    points = dd(int)
    r = graph_root = graph
    en_passant = 0

    for a_move in moves:
        if a_move:
            debug_this.move_now = a_move['num']
            for move, side in ((a_move['white'], 'W'), (a_move['black'], 'B')):
                if move:
                    mj = agn.parse(move[0]).unwrap()
                    if not isinstance(mj, list):
                        mj = [mj]
                    for m in mj:
                        sidemove = f"{a_move['num']}{side}"
                        print(f"[ep:{en_passant}]{debug_this.game_now}:{sidemove}.",
                              a_move[SIDE_REPR[side]][0],
                              json.dumps(m, sort_keys=True))

                        try:
                            points[side] += b.move(m, side)

                        except Exception as ee:
                            import traceback as tb
                            tb.print_exception(ee)
                            import pudb; pu.db
                            x = 1

                        if args.show_board and (
                            debug_this.in_game() or
                            debug_this.game_now >= debug_this._game_num):
                        # if args.show_board:  # and debug_this.in_game():
                            print(b, "\n")
                        print(f"W:{points['W']}, B:{points['B']}")
                            # b.redraw_gui_board()
                            # if debug_this._game_num is None and input().lower() == 'q':
                            #     exit()

            try:
                key = (a_move['white'][0],
                       a_move['black'][0] if a_move['black'] else None,
                       points['W'] - points['B'])
            except Exception as eee:
                import pudb; pu.db
                x = 1

            if key in graph:
                graph[key]['gcount'] += 1
                graph = graph[key]
            else:
                graph[key] = new_graph = dict(gcount=1)
                graph = new_graph
            # if args.show_board:
                # print(graph_root)
                # if debug_this._game_num is None and input().lower() == 'q':
                #     exit()

    """
    if move[0] in ograph:
        ograph[move[0]]['freq'] += 1
    else:
        ograph[move[0]] = {
            'freq': 1,
        }
    if mj['action'] == 'capture':
        import pudb;
        pu.db
        ograph[move[0]]['point'] = capture_points(mj)

    ograph[a_move['white'][0]] = a_move['white'][1]
    """


def handle_pgn(pgn, results: dict):
    result = pgn_file.parse(pgn)
    if isinstance(result, Success):
        result = result.unwrap()[0]
        results['gcount'] += 1
        board(result['game']['moves'], graph=results['ograph'])
        return

        for num, a_move in enumerate(result['game']['moves']):
            try:
                if a_move:
                    results['mcount'] += 1
                    print("a_move['white']", a_move['white'])
                    results['ograph'][a_move['white'][0]] = a_move['white'][1]
                    if a_move['white']:
                        if a_move['white'][0] in results["moves"]:
                            results["moves"][a_move['white'][0]] += 1
                        else:
                            results["moves"][a_move['white'][0]] = 1
                    else:
                        logger.error("empty white move! at {num} in {pgn}")
                    if a_move['black']:
                        if a_move['black'][0] in results["moves"]:
                            results["moves"][a_move['black'][0]] += 1
                        else:
                            results["moves"][a_move['black'][0]] = 1
                else:
                    logger.error("empty move! at {num} in {pgn}")
            except Exception as e:
                logger.exception(
                    "Exception whilst combining moves: %s", a_move, e)
    else:
        logger.error(f"Error parsing PGN: {pgn}")
        logger.error(result)


def pgn_worker(queue: Queue, queue_id: Text, process_count: int):
    logging.info("PROCESSQ: %s, %s -- %%%%%%%%%%%%",
                 str(queue), str(queue_id))
    work_count = 0
    results = dict({
        "gcount": 0, "mcount": 0, "moves": dict(), "ograph": dict()})
    while True:
        time.sleep(0.001)
        qsize = queue.qsize()
        if qsize:
            pgn = queue.get()
            if pgn is None:
                queue.put(json.dumps(results))
                break
            handle_pgn(pgn, results)
            work_count += 1
            if work_count % 500 == 0:
                logger.info("working: %s, %d, %d", queue_id, qsize, len(pgn))

        if results['gcount'] % 1000 == 0:
            count = results['gcount']
            this_time = time.time()
            time_per_k = (this_time - start_time) / (count / 1000)
            left = (todo / process_count) - count
            time_left = time_per_k * (left / 1000)
            logger.info(f"count: {count}, "
                          f"{round(time_per_k, 1)} sec/k, "
                          f"left: {left:,}, "
                          f"time left: {round(time_left / 3600, 2)} hrs, %s",
                          str(mp.current_process()))


def process_games_single(pgn_parser: PGNStreamSlicer, pgn_limit: int):
    results = dict({
        "gcount": 0, "mcount": 0, "moves": dict(), "ograph": dict()})
    for pgn_num, pgn in enumerate(pgn_parser.next()):
        debug_this.game_now = pgn_num
        handle_pgn(pgn, results)
        if pgn_limit and pgn_num > pgn_limit:
            break
    return results


def process_games_multi(pgn_parser: PGNStreamSlicer, process_count: int, pgn_limit: int, queue_limit: int):
    queues = dict()
    processes = []
    process_to_queue = dict()
    mgr = mp.Manager()
    for i in range(process_count):
        queue_id = f"queue_{i}"
        queues[queue_id] = this_q = mp.Queue()
        x = Process(
            target=pgn_worker,
            kwargs=dict(queue=this_q,
                        queue_id=queue_id,
                        process_count=process_count)
        )
        process_to_queue[x] = this_q
        processes.append(x)
        logger.debug("process created: %s, %s", str(this_q), str(x))

    logger.debug("Queues: %s", [", ".join(str(q) for q in queues)])
    started = False
    process_idx = 0
    queue_id = f"queue_{process_idx}"
    this_q = queues[queue_id]
    for pgn_num, pgn in enumerate(pgn_parser.next()):
        if pgn_num % 1000 == 0:
            logger.warning("pgn_num %d", pgn_num)
        if pgn_limit and pgn_num > pgn_limit:
            break
        loop_ctr = 0
        while True:
            if this_q.qsize() < queue_limit:
                this_q.put(pgn)
                break
            logger.info("Flipped the lid: %d, %d, %s", this_q.qsize(), pgn_num, process_idx)
            loop_ctr += 1
            process_idx = (process_idx + 1) % process_count
            queue_id = f"queue_{process_idx}"
            this_q = queues[queue_id]
            if loop_ctr > process_count:
                if not started:
                    for process in processes:
                        logger.debug("starting: %s", str(process))
                        process.start()
                    started = True
                logger.info("JAMMED UP - all queues full, waiting to clear"
                            "%d, %s, %d, %s",
                      pgn_num, process_idx, loop_ctr, "*"*80)
                time.sleep(1)

    for q in queues.values():
        q.put(None)

    merged = dict(gcount=0, mcount=0, moves=dd(int))
    for p in processes:
        p.join()
        results = json.loads(process_to_queue[p].get())
        merged['gcount'] += results['gcount']
        merged['mcount'] += results['mcount']
        for move, count in results['moves'].items():
            merged['moves'][move] += count

    mgr.shutdown()

    return merged


class GameDebug:
    def __init__(self, debug_at_game_move_csv=None):
        if debug_at_game_move_csv:
            self._game_num, self._move_num = (
                int(x) for x in debug_at_game_move_csv.split(","))
        else:
            self._game_num = None
            self._move_num = None

        self._game_now = None
        self._move_now = None

    @property
    def game_now(self):
        return self._game_now

    @game_now.setter
    def game_now(self, game_now: int):
        self._game_now = game_now

    @property
    def move_now(self):
        return self._move_now

    @move_now.setter
    def move_now(self, move_now: int):
        self._move_now = move_now

    def now(self, margin=0):
        # Have we reached the game and move # desired?
        return (self._game_num is not None and self._move_num is not None and
                self._game_num == self._game_now and
                self._move_num <= self._move_now <= self._move_num +1)

    def in_game(self):
        return self._game_num is not None and self._game_num == self._game_now

    def in_use(self):
        return self._game_num is not None

    def __repr__(self):
        return (f"expected: {self._game_num},{self._move_num} " +
                ("==" if self.now() else "!=") +
                f" current: {self._game_now},{self._move_now}")


debug_this: GameDebug = None


def main(argv):
    global debug_this
    args, args_parser = get_args(argv)
    parser = PGNStreamSlicer(pgn_path=args.pgn_path)

    # if args.game_move_debug is not None:
    debug_this = GameDebug(args.game_move_debug)

    if args.process_count > 1:
        move_stats = process_games_multi(parser, args.process_count, args.pgn_limit, args.queue_limit)
    else:
        move_stats = process_games_single(parser, args.pgn_limit)

    """
    with open(args.out_path, "w") as f:
        f.write(json.dumps({
            "game_count": move_stats['gcount'],
            "move_count": move_stats['mcount'],
            "moves": sorted(move_stats['moves'].keys()),
            "stats": {k: v for k, v in move_stats.items()},
        })+"\n\n")
    """


if __name__ == "__main__":
    main(sys.argv[1:])
