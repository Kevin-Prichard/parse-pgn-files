#!/usr/bin/env python3
#cython: language_level=3

import argparse
import copy
from collections import defaultdict
from enum import Enum
import json
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import re
import sys
import time
from typing import List, Tuple, Callable, Text, Dict, Union

from common import OPPO, PINS_RX, CHECKS_RX
from pgn_parser import pgn_file
from move_parser import agn, FILE_CHARS

from box import Box
from colored import fore, back
from parsita import Success
import pudb

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


PAWN_ACTION = {
    'W': {'start': 1, 'first': [2, 3], 'advance': 1, 'capture': 1, 'promote': 6},
    'B': {'start': 6, 'first': [5, 4], 'advance': -1, 'capture': -1, 'promote': 1}
}
EMPTY_PIN = tuple()



class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()


args = None
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
    parser.add_argument('--quick', '-k', dest='quick_skip',
                        action='store_true', default=False,
                        help="Skip to the game:move indicated by --debug")

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
}
COLOR_SCHEME_DEFAULT = 'blue_purple'
PIECE_STYLE_DEFAULT = 'solid'

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
        # Saw off PGN hunks and yield them to caller
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
promotions = 0

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


class IsChecked:
    pass


class IsPinned:
    pass


class AmbiguousMoveError(Exception):
    pass

BISHOP_DELTAS = (
    (-1, 1), (1, 1),
    (1, -1), (-1, -1))

QUEEN_DELTAS = (
    (-1, 1), (0, 1), (1, 1),
    (-1, 0),           (1, 0),
    (-1, -1), (0, -1), (1, -1))

ROOK_DELTAS = (
    (-1, 0), (0, 1),
    (1, 0), (0, -1))


def eval_king_state(piece, square, board, side,
                    return_result=False,
                    edits: Union[Dict, None]=None):
    """
    This is a post-move check, determines whether a Q|R|B piece's new location
    is pinning an opponent piece to the opponent king
    """
    match piece:
        case 'B':
            deltas = BISHOP_DELTAS
        case 'R':
            deltas = ROOK_DELTAS
        case 'Q':
            deltas = QUEEN_DELTAS
        case _:
            raise ValueError(f"Unexpected piece type: {piece}")

    start = mk_coords(square)
    opking_sq = board.by_piece('K', OPPO[side])[0]
    if not opking_sq:
        raise ValueError(f"Opponent's king not found: {OPPO[side]}")
    opking = mk_coords(opking_sq)
    kdx, kdy = opking[0] - start[0], opking[1] - start[1]

    for dx, dy in deltas:
        keep_on = True

        # could this be a pin line?
        is_pin_line = (
            ((kdx == 0 and dx == 0) or (kdx and (kdx / abs(kdx))) == dx) and
            ((kdy == 0 and dy == 0) or (kdy and (kdy / abs(kdy))) == dy))

        if not is_pin_line:
            continue

        # starting point is current piece square
        new_file, new_rank = start[0], start[1]

        if edits and square in edits:
            pin_stripe = [edits[square] or "  "]
        else:
            pin_stripe = [board.get_coords(new_rank, new_file) or "  "]

        pinned_piece = None
        pinned_sq = None

        # keep seaching along the radial until out-of-bounds
        while keep_on:
            new_file += dx
            new_rank += dy

            # If out-of-bounds this radial search ends here
            if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                break

            # get text representation of piece at new square
            a_square = mk_square(new_file, new_rank)
            if edits and a_square in edits:
                piece_at = edits[a_square]
            else:
                piece_at = board.get_coords(file=new_file, rank=new_rank)

            if piece_at:
                if piece_at[1] == OPPO[side]:
                    if piece_at[0] == 'K':
                        # two opponent pieces on this line: no pin
                        keep_on = False
                    elif not pinned_piece:
                        pinned_piece = piece_at
                        pinned_sq = mk_square(new_file, new_rank)
                    else:
                        is_pin_line = False
                        keep_on = False


            pin_stripe.append(piece_at or "  ")

        if is_pin_line and pin_stripe:  # and pinned_piece:
            # is opponent's king pinned?
            try:
                stripe = "".join(pin_stripe)
                if PINS_RX[side].match(stripe):
                    logger.debug(
                        f"PIN FOUND: {piece}{side} at {square} pins "
                        f"{pinned_piece} at {pinned_sq} to {opking_sq} "
                        f"{'RP' if return_result else '-'}")
                    if return_result:
                        return IsPinned, (pinned_piece, pinned_sq, opking_sq)
                    board.add_pin(pinned_piece,
                                  pinned_sq,
                                  f"{piece}{side}",  # pinning piece
                                  square,            # pinning square
                                  opking_sq)         # opponent king square
                    return IsPinned
                elif CHECKS_RX[side].match(stripe):
                    if return_result:
                        return IsChecked, (f"{piece}{side}", square, opking_sq)
                    board.add_check(f"{piece}{side}", square, opking_sq)
                    return IsChecked
            except Exception as ee:
                import traceback as tb
                print(tb.format_exc())
                print(f"Error: {ee}")
                pu.db
                x = 1

    return None


def check_for_king_pin(board: 'Board', side):
    # Check oppo's sliders for pinning side's king
    for piece_type in "QRB":
        for square in board.by_piece(piece_type, OPPO[side]):
            eval_king_state(piece_type, square, board, OPPO[side])


def dests_radial(piece, square, radial_deltas, action, board, side):
    # TODO: combine this with check_for_pin, or at least seed it with the
    # deltas that align with the king
    """
    piece movement destinations based on radial delta paths
    e.g. queen moves along eight radial paths relative to current square
    Important: the in-memory board's origin is a1 aka board.board[0][0]
    which is the lower-left corner. coord_alters pairs are relative to this.
    """
    moves = set()

    start = mk_coords(square)
    opking = mk_coords(board.by_piece('K', OPPO[side])[0])

    kdx, kdy = start[0] - opking[0], start[1] - opking[1]

    # dx,dy is the change at each step of the radial
    for dx, dy in radial_deltas:
        keep_on = True

        # starting point is current piece square
        new_file, new_rank = start[0], start[1]

        # keep seaching along the radial until out-of-bounds
        while keep_on:
            new_file += dx
            new_rank += dy

            # out-of-bounds ends this radial search
            if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                break

            # get text representation of piece at new square
            piece_at = board.get_coords(file=new_file, rank=new_rank)
            # if is_pin_line:
            #     pin_stripe.append(piece_at or "  ")

            if action == 'capture':
                if piece_at:
                    if piece_at[1] == OPPO[side]:
                        moves.add(mk_square(new_file, new_rank))
                    # either way, either side, this radial ends here
                    keep_on = False

            elif action == 'move':
                if not piece_at:
                    moves.add(mk_square(new_file, new_rank))
                else:
                    # piece collision ends this radial, either side
                    keep_on = False

    return moves


def dests_knight(square):
    return dests_common(square,
                        ((-2, 1), (-1, 2), (1, 2), (2, 1),
                         (-2, -1), (-1, -2), (1, -2), (2, -1)))


def dests_king(square):
    return dests_common(square,
                        ((-1, 1,), (0, 1), (1, 1),
                         (-1, 0),           (1, 0),
                         (-1, -1), (0, -1), (1, -1)))


def dests_bishop(square, action, board, side):
    # Clockwise from 10:30
    return dests_radial("B", square, BISHOP_DELTAS,
                        action=action, board=board, side=side)


def dests_rook(square, action, board, side):
    # Clockwise from 9:00
    return dests_radial("R", square, ROOK_DELTAS,
                        action=action, board=board, side=side)

def dests_queen(square, action, board, side):
    # clockwise from 10:30
    return dests_radial("Q", square, QUEEN_DELTAS,
                        action=action, board=board, side=side)


def dests_pawn(square, side, action, board=None, target=None):
    dests = set()
    start = mk_coords(square)

    capture_dir = None
    act = PAWN_ACTION[side]
    if action == 'move':
        # move
        if start[1] == act['start']:
            if not board.get(tsq := mk_square(
                    start[0], act['first'][0])):
                dests.add(tsq)
                if not board.get(tsq := mk_square(
                        start[0], act['first'][1])):
                    dests.add(tsq)
        else:
            if not board.get(tsq := mk_square(
                    start[0], start[1] + act['advance'])):
                dests.add(tsq)

    if action == 'capture':
        if start[0] > 0:
            if board.get(tsq := mk_square(
                    start[0] - 1, start[1] + act['capture'])):
                dests.add(tsq)
        if start[0] < 7:
            if board.get(tsq := mk_square(
                    start[0] + 1, start[1] + act['capture'])):
                dests.add(tsq)

    if action == 'promote':
        if side == 'W' and start[1] >= act['promote']:
            dests.add(mk_square(start[0], 7))
        elif side == 'B' and start[1] <= act['promote']:
            dests.add(mk_square(start[0], 0))

    return dests


class Board:
    def __init__(self, empty=False):
        self._pieces = defaultdict(set)
        if not empty:
            self.board = copy.deepcopy(BOARD_STD)
            self._populate_pieces()
        else:
            self.board = [[None] * 8 for _ in range(8)]
        self._pinned = dict()
        self._pinner = dict()
        self._king_pin = defaultdict(set)
        self._checks = defaultdict(set)
        # self._pinner = dict()
        # self._gui_board, self._gui_board_window = prepare_board()

    def _populate_pieces(self):
        for rank in (0, 1, 6, 7):
            for file in range(8):
                piece = self.board[rank][file]
                if piece:
                    self.add(piece, mk_square(file, rank))

    def add(self, piece, square):
        self._pieces[piece].add(square)
        coords = mk_coords(square)
        self.board[coords[1]][coords[0]] = piece

    def add_check(self, piece, square, king_sq):
        self._checks[king_sq].add((piece, square))

    def is_checked(self, king_sq, side):
        czech_piece, czech_sq = self._checks.get(king_sq)
        is_checked, check_info = eval_king_state(
            piece=czech_piece, square=czech_sq, board=self,
            side=side, return_result=True, edits=None)
        if is_checked is not IsChecked:
            return False
        return check_info

    def add_pin(self, pinned_piece, pinned_sq, pinning_piece, pinning_sq, king_sq):

        # A piece can pin only a single piece to opponent king
        if ((pinning_piece, pinning_sq) in self._pinner and
                self._pinner[pinning_piece, pinning_sq] !=
                (pinned_piece, pinned_sq, king_sq)):
            logger.debug(
                f"REPLACING PINNED PIECE: "
                f"{self._pinner[pinning_piece, pinning_sq]} "
                f"with {(pinned_piece, pinned_sq, king_sq)}")
        self._pinner[pinning_piece, pinning_sq] = (pinned_piece, pinned_sq, king_sq)

        # A piece can be pinned by only one opponent piece at a time
        if ((pinned_piece, pinned_sq) in self._pinned and
                self._pinned[pinned_piece, pinned_sq] !=
                (pinning_piece, pinning_sq, king_sq)):
            pu.db
        self._pinned[pinned_piece, pinned_sq] = (pinning_piece, pinning_sq, king_sq)

        # A king can have more than one piece pinned to it
        self._king_pin[king_sq].add((pinned_piece, pinned_sq, pinning_piece, pinning_sq))

    def is_pinned(self, piece, side, square) -> [None, Tuple[str, str, str]]:
        return self._pinned.get((f"{piece}{side}", square))

    def review_pins(self):
        tasks = []
        # Review pinners to see if they are still pinning
        for pinner, pinfo in self._pinner.items():
            # If a remembered pinner isn't pinning, queue to remove the pin
            try:
                result = eval_king_state(
                    pinner[0][0], pinner[1], self,
                    pinner[0][1], return_result=True)
                if result is not None:
                    is_pinned, new_pinfo = result
                if result is None or is_pinned is not IsPinned or not new_pinfo or new_pinfo != pinfo:
                    tasks.append((pinner, pinfo))
            except Exception as eee:
                pu.db
                x = 1
        # Must defer removal of pins until all pins have been reviewed,
        # because that would interrupt iteration of dict self._pinner
        for pinner, pinfo in tasks:
            try:
                if type(pinfo) == bool:
                    pu.db
                logger.debug(
                    f"PIN REMOVED*** {pinner[0]} at {pinner[1]} pins "
                    f"{pinfo[0]} at {pinfo[1]} to {pinfo[2]}")

                # Remove pinner
                del self._pinner[pinner]

                # Remove pinned piece
                if pinfo[:2] not in self._pinned:
                    pu.db
                del self._pinned[pinfo[:2]]

                # Remove king pin if no longer pinned against, or no longer on that space
                if pinfo[2] not in self._king_pin:
                    pu.db
                self._king_pin[pinfo[2]].remove((*pinfo[:2], *pinner))
                if not self._king_pin[pinfo[2]]:
                    del self._king_pin[pinfo[2]]

            except Exception as ee:
                import traceback as tb
                print(tb.format_exc())
                pu.db
                x = 1

    def review_checks(self):
        """
        The point of review_checks() is to
        1) Remove checks that are no longer
        2) Turn checks into pins if the checked side moves a piece to block
           the check, whereupon that piece is pinned to the king
        """
        tasks = []
        for king_sq, checks in self._checks.items():
            for check in checks:
                is_checked = eval_king_state(
                    piece=check[0][0], square=check[1], board=self,
                    side=check[0][1], return_result=False)
                if is_checked is not IsChecked:
                    tasks.append((king_sq, check))

        for king_sq, check in tasks:
            self._checks[king_sq].remove(check)
            logger.debug(f"CHECK REMOVED%%%: {check}")
            if not self._checks[king_sq]:
                del self._checks[king_sq]

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

            if self._pinned:
                self.review_pins()
            if from_piece[0] == 'K':
                check_for_king_pin(self, from_piece[1])
            elif from_piece[0] in "QRB":
                eval_king_state(from_piece[0], to_sq, self, from_piece[1])
            if self._checks:
                self.review_checks()
            if self._pinned or self._pinner or self._king_pin or self._checks:
                logger.debug(
                    f"*Pinners: {self._pinner}\n"
                    f"*Pinned: {self._pinned}\n"
                    f"*King pin: {self._king_pin}\n"
                    f"*Checks: {self._checks}")

        except Exception as ee:
            import traceback as tb
            print(tb.format_exc())
            import pudb; pu.db
            x = 1

    def redraw_gui_board(self):
        for rank in range(8):
            for file in range(8):
                self._gui_board.psg_board[rank][file] = (
                    PSG)[self.get_coords(rank, file)]
        self._gui_board.redraw_board(self._gui_board_window)

    def get(self, square) -> Union[str|None]:
        return self.board[int(square[1])-1][ord(square[0])-FILE_BASE]

    def get_coords(self, rank, file):
        return self.board[rank][file]

    def set(self, square, piece=None, side=None):
        try:
            self.board[int(square[1])-1][ord(square[0])-FILE_BASE] = \
                f"{piece}{side}" if piece else None
        except Exception as ee:
            print(self)
            import pudb; pu.db
            x = 1

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

    def move(self, m, side):
        global en_passant, promotions
        points = 0
        old, new = None, None
        if m['action'] == 'promote':
            promotions += 1

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
            potentials = set()
            candidates = self.by_file(m['target'][0], m['piece'], side)
            for piece_square in candidates:
                if m['target'] in self.piece_dests(
                        piece=m['piece'], start_square=piece_square,
                        side=side, action=m['action'], board=self):
                    potentials.add((piece_square, m['target']))

            if len(potentials) == 1:
                # We set old and new to avoid error at this method's end
                # action, because action == 'promote' is ignored
                old, new = potentials.pop()
            else:
                raise AmbiguousMoveError(
                    f"Multiple possible moves for {m}:\n{potentials}")
            self.set(old, piece=None)
            self.set(new, piece=m['replacement'], side=side)
            self._pieces[f"{m['piece']}{side}"].remove(old)
            self._pieces[f"{m['replacement']}{side}"].add(new)

        elif 'actor_file' in m:
            squares = self.by_file(m['actor_file'], m['piece'], side)

            if m['action'] == 'promote':
                if len(squares) != 1:
                    pu.db

            # Iterate current origin squares of matching pieces
            for piece_square in squares:

                # Is the current move's destination in one of the piece's
                # possible destinations?
                candidates = self.piece_dests(
                        piece=m['piece'], start_square=piece_square,
                        side=side, action=m['action'], board=self)

                if m['action'] == 'promote':
                    pu.db
                    old = piece_square
                    new = candidates[0]
                    break
                elif m['target'] in candidates:
                    if pinner := self.is_pinned(m['piece'], side, piece_square):
                        if m['action'] != 'capture':
                            continue
                        cap_target = self.get(m['target'])
                        if cap_target == pinner[0]:
                            points += POINTS[cap_target[0]]
                    old = piece_square
                    new = m['target']
                    break
            if old is None or new is None:
                raise AmbiguousMoveError(
                    f"Moving piece isn't found: {m}")

        elif 'actor_rank' in m:
            squares = self.by_rank(m['actor_rank'], m['piece'], side)
            # Iterate current origin squares of matching pieces
            for piece_square in squares:

                # Is the current move's destination in one of the piece's
                # possible destinations?
                if m['target'] in self.piece_dests(
                        piece=m['piece'], start_square=piece_square,
                        side=side, action=m['action'], board=self):
                    if not self.is_pinned(m['piece'], side, piece_square):
                        old = piece_square
                        new = m['target']
                        break
            if old is None or new is None:
                raise AmbiguousMoveError(
                    f"Moving piece isn't found: {m}")

        elif 'actor_square' in m and not self.is_pinned(m['piece'], side, m['actor_square']):
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

            potentials = set()
            # Iterate current origin squares of matching pieces
            for idx, piece_square in enumerate(squares):

                # Is the current move's destination in one of the piece's
                # possible destinations?
                if (m['piece'] == 'P' and 'target' in m
                        and (m['action'] == 'move' and piece_square[0] != m['target'][0])):
                    continue
                candidates = self.piece_dests(
                    piece=m['piece'], start_square=piece_square,
                    side=side, action=m['action'], board=self,
                    target=m['target'])

                if m['target'] in candidates:
                    did_add = False
                    if not (pinner := self.is_pinned(m['piece'],
                                                     side, piece_square)):
                        potentials.add((m['piece'], piece_square, m['target']))
                        did_add = True
                    else:
                        # the piece currently is pinned pre-move;
                        # does this move maintain the pin?
                        # Peer into the future by applying the move to board state
                        result = eval_king_state(
                            pinner[0][0], pinner[1], self,
                            pinner[0][1], return_result=True,
                            edits={m['target']: f"{m['piece']}{side}",
                                   piece_square: None})
                        if result is None or result[0] is IsPinned:
                                              #result[0] is not IsChecked:
                            potentials.add((m['piece'], piece_square, m['target']))
                            did_add = True

            if len(potentials) == 1:
                old, new = potentials.pop()[1:]
            else:
                raise AmbiguousMoveError(
                    f"Multiple possible moves for {m}:\n{potentials}")
            if old is None or new is None:
                pu.db
                x = 1

        if old is None or new is None:
            print(self)
            pu.db
            raise ValueError(f"Moving piece isn't found: {m}")

        if m['action'] == 'capture':
            cap_target = self.get(m['target'])
            points += POINTS[cap_target[0]] if cap_target else 0

        if m['action'] != 'promote':
            self.reposition(old, new, is_capture=m['action'] == 'capture')

        return points

    def __repr__(self):
        res = []
        scheme = args.colors if args else COLOR_SCHEME_DEFAULT
        colors = COLOR_SCHEMES[scheme] if scheme != 'none' else None
        style = args.piece_style if args else PIECE_STYLE_DEFAULT
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq_col = int(not((rank + file) % 2))
                piece = self.board[rank][file]
                if colors:
                    row.append(
                        f"{back(colors[SQUARE_COL[sq_col]])}"
                        f"{fore(colors[PIECE_COL[piece[1] if piece else 'B']])}"
                        f"{DISPLAY[style][piece[0]] if piece else ' ':1s}")
                else:
                    row.append(
                        f"{DISPLAY[style][piece[0]] if piece else ' ':1s}")
                sq_col = int(not(sq_col))
            if colors:
                row.append(f"{back('black')}{fore('white')}")
            res.append(" ".join(row))
        return "\n".join(res)


ambiguous_game_count = 0

def run_game(moves, graph):
    global en_passant, ambiguous_game_count
    b = Board()
    points = defaultdict(int)
    r = graph_root = graph
    en_passant = 0
    if debug_this.game_now % 100 == 0:
        logger.warning("Game: %d", debug_this.game_now)

    for a_move in moves:
        if a_move:
            debug_this.move_now = a_move['num']
            for move, side in ((a_move['white'], 'W'), (a_move['black'], 'B')):
                if move:
                    try:
                        mj = agn.parse(move[0]).unwrap()

                    except Exception as e:
                        import traceback as tb
                        print(tb.format_exc())
                        print(f"Error parsing move: {move}")
                        pu.db
                        x = 1

                    if not isinstance(mj, list):
                        mj = [mj]
                    for m in mj:
                        sidemove = f"{a_move['num']}{side}"
                        move_header = (
                            f"[ep:{en_passant}]{debug_this.game_now}:{sidemove}."
                            f"{a_move[SIDE_REPR[side]][0]} ")
                        logger.debug(move_header +
                                     json.dumps(m, sort_keys=True))
                        try:
                            points[side] += b.move(m, side)

                        except AmbiguousMoveError as ame:
                            ambiguous_game_count += 1
                            logger.error(
                                f"Ambiguous move {move[0]}, "
                                f"abandoning game: "
                                f"{debug_this.game_now}:{sidemove}; "
                                f"games abandoned: {ambiguous_game_count}")
                            return

                        except Exception as ee:
                            import traceback as tb
                            tb.print_exception(ee)
                            import pudb; pu.db
                            x = 1

                        if args.show_board and debug_this.in_game(True):
                            # if args.show_board:  # and debug_this.in_game():
                            print(b, "\n")
                        logger.debug(f"W:{points['W']}, B:{points['B']}")
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


def handle_pgn(pgn, results: dict):
    result = pgn_file.parse(pgn)
    if isinstance(result, Success):
        result = result.unwrap()[0]
        results['gcount'] += 1
        run_game(result['game']['moves'], graph=results['ograph'])
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
        if args.quick_skip and pgn_num < debug_this._game_num:
            continue
        else:
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

    merged = dict(gcount=0, mcount=0, moves=defaultdict(int))
    for p in processes:
        p.join()
        results = json.loads(process_to_queue[p].get())
        merged['gcount'] += results['gcount']
        merged['mcount'] += results['mcount']
        for move, count in results['moves'].items():
            merged['moves'][move] += count

    mgr.shutdown()

    return merged


DEBUG_SPLIT_RX = re.compile(r"^(?P<game_num>\d+)\D(?P<move_num>\d+)$")

class GameDebug:
    def __init__(self, debug_at_game_move_csv=None):
        if (debug_at_game_move_csv and
                (m := Box(DEBUG_SPLIT_RX.match(
                    debug_at_game_move_csv).groupdict()))):
            self._game_num, self._move_num = int(m.game_num), int(m.move_num)
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

    def in_game(self, gt=None):
        return ((self._game_num is not None and
                 self._game_num == self._game_now) or (
                        gt is not None and
                        self._game_num is not None and
                        self._game_now >= self._game_num)
        )


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

    with open(args.out_path, "w") as f:
        f.write(json.dumps({
            "game_count": move_stats['gcount'],
            "move_count": move_stats['mcount'],
            "moves": sorted(move_stats['moves'].keys()),
            "stats": {k: v for k, v in move_stats.items()},
        })+"\n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
