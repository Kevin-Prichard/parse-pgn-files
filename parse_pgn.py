#!/usr/bin/env python

import argparse
import json
from collections import defaultdict as dd
from enum import Enum
import logging
import re
import sys
from typing import List, Tuple, Callable, Text, Dict

from pgn_parser import pgn_file


logger = logging.getLogger(__name__)


NAUGHTY_CHARS = re.compile(b'[^\n -~]+')


class PGNParts(Enum):
    ANNOTATION = 1
    MOVES = 2
    NEWLINE = 3


# PGN parts can be identified by the first character of the line,
# which helps break a PGN file into its constituent parts.
LINE_TYPE = {
    b'[': PGNParts.ANNOTATION,
    b'1': PGNParts.MOVES,
    b'\n': PGNParts.NEWLINE,
}


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()


def get_args(argv: List[str]) -> Tuple[
        argparse.Namespace, ArgumentParser]:
    parser = ArgumentParser(
        prog='./parse_pgn.py',
        description='Parse PGN file(s) and stream(s): '
                    'parse PGNs and pass the resulting '
                    'parse tree to a function.')
    parser.add_argument('--file', '-f', dest='pgn_path',
                        type = str, action = 'store', default = '-')
    parser.add_argument('--output', '-o', dest='out_path',
                        type = str, action = 'store', default = '/dev/stdout')
    return parser.parse_args(argv), parser


class PGNParser:
    def __init__(self, pgn_path: str, parse_cb: Callable[[Dict], None] = None):
        self.pgn_path = pgn_path
        self.pgn_file = open(self.pgn_path, 'rb')  # PGNs are UTF-8 encoded, e.g. "Réti Opening"
        self.parse_cb = parse_cb
        self.result = None

    def get_pgn(self):
        # Saw off PGN hunks
        buf = []
        this_type, prev_type = None, None
        count = 0
        while line := self.pgn_file.readline():
            this_type = LINE_TYPE.get(line[:1])
            if ( this_type == PGNParts.ANNOTATION and
                    prev_type == PGNParts.MOVES ):
                final = b"\n".join(buf)
                # When annotations contain non-ASCII characters, wipe them out
                # because parsita doesn't handle them well.
                final_utf8 = NAUGHTY_CHARS.sub(
                    b'_', final, 999).decode('utf-8')
                yield final_utf8
                count += 1
                if count % 1000 == 0:
                    print(f"count: {count}")
                buf.clear()
                buf = [line[:-1]]
            else:
                buf.append(line[:-1])

            if this_type is not PGNParts.NEWLINE:
                prev_type = this_type

    @classmethod
    def parse(cls, pgn: Text):
        return pgn_file.parse(pgn)

    def parse_stream(self):
        for pgn in self.get_pgn():
            # print(pgn,"\n\n")
            result = pgn_file.parse(pgn)  # PGNParser.parse(pgn)
            if isinstance(result, Success):
                result = result.unwrap()
                if self.parse_cb:
                    self.parse_cb(result)
                else:
                    # There should only be one PGN per result, as get_pgn()
                    # segments the file or stream into single PGNs.
                    yield result[0]
            else:
                logger.error(f"Error parsing PGN: {pgn}")
                logger.error(result)


def main(argv):
    args, parser = get_args(argv)
    parser = PGNParser(pgn_path=args.pgn_path)
    movestr = dd(int)
    mc = 0
    for pgn in parser.parse_stream():
        for num, a_move in enumerate(pgn['game']['moves']):
            try:
                if a_move:
                    mc += 1
                    if a_move['white']:
                        movestr[a_move['white'][0]] += 1
                    else:
                        logger.error("empty white move! at {num} in {pgn}")
                    if a_move['black']:
                        movestr[a_move['black'][0]] += 1
                else:
                    logger.error("empty move! at {num} in {pgn}")
            except Exception as e:
                import pudb; pu.db
                x = 1
    with open(args.out_path, "w") as f:
        f.write(f"Moves {mc}\n")
        f.write(f"AN: {sorted(movestr.keys())}\n\n")
        f.write(json.dumps(movestr, indent=4))


if __name__ == '__main__':
    main(sys.argv[1:])
