#!/usr/bin/env python

import argparse
from enum import Enum
import re
import sys
from typing import List, Tuple, Callable, Text

from parsita import *


naughty_chars = re.compile(b'[^\n -~]+')


class PGNPart(Enum):
    ANNOTATION = 1
    MOVES = 2
    NEWLINE = 3


LINE_TYPE = {
    b'[': PGNPart.ANNOTATION,
    b'1': PGNPart.MOVES,
    b'\n': PGNPart.NEWLINE,
}


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()


def get_args(argv: List[str]) -> Tuple[
        argparse.Namespace, argparse.ArgumentParser]:
    parser = ArgumentParser(
        prog='./parse_pgn.py',
        description='Parse PGN file(s) and stream(s): '
                    'parse PGNs and pass the resulting '
                    'parse tree to a function.')
    parser.add_argument('--file', '-f', dest='pgn_path',
                        type = str, action = 'store', default = '-')
    return parser.parse_args(argv), parser


class PGNParser:
    def __init__(self, pgn_path: str, parse_cb: Callable[[str], None] = None):
        self.pgn_path = pgn_path
        self.pgn_file = open(self.pgn_path, 'rb')
        self.parse_cb = parse_cb
        self.result = None

    def get_pgn(self):
        buf = []
        this_kind, prev_kind = None, None
        count = 0
        while line := self.pgn_file.readline():
            # line = self.pgn_file.readline()
            this_kind = LINE_TYPE.get(line[:1])
            if this_kind == PGNPart.ANNOTATION and prev_kind == PGNPart.MOVES:
                final = b"\n".join(buf)
                final_utf8 = naughty_chars.sub(b'_', final, 999).decode('utf-8')
                import pudb; pu.db
                yield final_utf8
                print(f"count: {count}")
                count += 1
                buf.clear()
                buf = [line[:-1]]
            else:
                buf.append(line[:-1])

            if this_kind is not PGNPart.NEWLINE:
                prev_kind = this_kind

    @classmethod
    def parse(cls, pgn: Text):

        # if naughty_chars.match(pgn):
        #     import pudb; pu.db

        return cls.pgn_file.parse(pgn)

    def parse_stream(self):
        for pgn in self.get_pgn():
            r = self.result = PGNParser.parse(pgn)
            import pudb; pu.db
            if self.parse_cb:
                self.parse_cb(self.result)

    def formatannotations(annotations):
        return {ant[0]: ant[1] for ant in annotations}

    def formatgame(game):
        return {
            'moves': game[0],
            'outcome': game[1]
        }

    def formatentry(entry):
        return {'annotations': entry[0], 'game': entry[1]}

    def handleoptional(optionalmove):
        if len(optionalmove) > 0:
            return optionalmove[0]
        else:
            return None

    # Define Grammar by building up from smallest components

    # tokens
    quote = lit(r'"')
    whitespace = lit(' ') | lit('\n')
    tag = reg(r'[\u0021-\u0021\u0023-\u005A\u005E-\u007E]+')
    string = reg(r'[\u0020-\u0021\u0023-\u005A\u005E-\U0010FFFF]+')

    # Annotations: [Foo "Super Awesome Information"]
    annotation = "[" >> tag << " " & (quote >> string << quote) << "]"
    annotations = repsep(annotation, '\n') > formatannotations

    # Moves are more complicated
    regularmove = reg(
        r'[a-h1-8NBRQKx\+#=]+')  # Matches more than just chess moves
    longcastle = reg(
        r'O-O-O[+#]?')  # match first to avoid castle matching spuriously
    castle = reg(r'O-O[+#]?')
    nullmove = lit('--')  # Illegal move rarely used in annotations

    # LiChess annotations
    moveAnnotation = reg(r'\{[^}]*\}')

    # Possible move types
    move = (regularmove | longcastle | castle | nullmove)

    # Build up the game
    movenumber = (reg(r'[0-9]+') << '.' << whitespace) > int
    turn = movenumber & (move << whitespace & (opt(moveAnnotation << whitespace))) & (
                opt(move << whitespace & (opt(moveAnnotation << whitespace))) > handleoptional)

    draw = lit('1/2-1/2')
    white = lit('1-0')
    black = lit('0-1')
    outcome = draw | white | black

    game = (rep(turn) & outcome) > formatgame

    # A PGN entry is annotations and the game
    entry = ((annotations << rep(whitespace)) & (
                game << rep(whitespace))) > formatentry

    # A file is repeated entries
    pgn_file = rep(entry)


def main(argv):
    args, parser = get_args(argv)
    parser = PGNParser(pgn_path=args.pgn_path)
    parser.parse_stream()


if __name__ == '__main__':
    a = main(sys.argv[1:])
    import pudb; pu.db
    x = 1
