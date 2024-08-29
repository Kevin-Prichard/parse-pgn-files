#!/usr/bin/env python3

import logging
import time

from parsita import lit, Failure

"""
Algebraic Notation grammar-parser
"""


def square_fmt(square):
    return square[0] + square[1]


def locator_fmt(locator):
    return locator[0] + locator[1]


def move_singular(content):  # singular piece move formatter
    return {"piece": content[0],
            "action": "move",
            "target": content[1]}


def move_plural(content):  # plural piece move formatter
    res = {"piece": content[0],
           "action": "move",
           "target": content[2]}
    if len(content[1]) > 1:
        res.update(actor_square=content[1])
    if content[1] in file_chars:
        res.update(actor_file=content[1])
    elif content[1] in rank_chars:
        res.update(actor_rank=content[1])
    return res


def move_castle(content):
    return {"piece": "K",
            "action": "move",
            "target": content
            }


def take_by_pawn(content):
    return {"piece": "P",
            "actor_file": content[0],
            "action": "capture",
            "target": content[2]}


def take_singular(content):
    return {"piece": content[0],
            "action": "capture",
            "target": content[2]}


def take_plural(content):
    res = {"piece": content[0],
           "action": "capture",
           "target": content[3]}
    if content[1] in file_chars:
        res.update(actor_file=content[1])
    elif content[1] in rank_chars:
        res.update(actor_rank=content[1])
    return res


def conversion_fmt(content):
    if content[1] == '=':
        return {
            "piece": "P",
            "actor_square": content[0],
            "replacement": content[2],
        }
    elif content[1] == 'x':
        return {
            "piece": "P",
            "actor_file": content[0],
            "target": content[2],
            "replacement": content[4],
        }
    logger.error("Did not handle pawn conversion: {content}", {"content": content})


logger = logging.getLogger(__name__)


piece = lit('P', 'N', 'B', 'R', 'Q', 'K')
plural = lit('P', 'N', 'B', 'R', 'Q')

castle = lit('O-O-O', 'O-O')
take = lit('x')
convert = lit('=')

file_chars = "abcdefgh"
file = lit(*list(file_chars))

rank_chars = "12345678"
rank = lit(*list(rank_chars))

locator = file | rank
square = file & rank > square_fmt

# There are approximately the following possible move types (& probably more)-
# Pawn moves: 8**2 [64]
# Pawn takes: 8**2 [64]
# Singular piece moves: 6 Pieces to 8**2 squares [384] (Nf3, Bb5, Rd1, Qe2, Kf1, Kg1)
# Singular piece takes: 6 Pieces to 8**2 squares [384] (Nxf3, Bxb5, Rxd1, Qxe2, Kxf1, Kxg1)
# Plural piece moves: 5 pieces (sans king) * 16 rank or file locators, to 8**2 squares [5120] (Ncf3, Bbf3, R8d1, Q4e2, Kg2, Kf1)
# Plural piece takes: 5 pieces (sans king) * 16 rank or file locators, to 8**2 squares [5120] (Ncxf3, Bbxf3, R8xd1, Q4xe2)
# Castling moves: 2
# Total: 64 + 384 + 384 + 5120 + 5120 + 2 = 10810 possible moves.  I think.

move = (square  # pawn
        | (piece & square > move_singular)  # singular piece
        | (plural & locator & square > move_plural)  # plural piece, from locator
        | (plural & square & square > move_plural)  # plural piece, from square
        | (castle > move_castle)  # castling
        )

capture = ((file & take & square > take_by_pawn)  # pawn
           | (piece & take & square > take_singular)  # singular piece
           | (plural & locator & take & square > take_plural)  # plural piece
           )

conversion = (square & convert & plural) > conversion_fmt
take_conversion = (file & take & square & convert & plural) > conversion_fmt

agn = move | capture | conversion | take_conversion

"""
[X] 1a. Get this parser working!
[X] 1b. Make it yield usable structures
2a. Eventually look into adding helper classes to parsita to transform a grammar to another grammar syntax, such as ANTLR or yacc etc
2b. This to produce a faster-executing version of a parser
3a. In the meantime, split up mega tasks using threads or multitasking, utilising concurrent dicts and structures 
3b. Consider also mypy and other python "compilers"
"""


def main():
    from result import AN
    t0 = time.time()
    for mv in AN:
        try:
            parsed = agn.parse(mv)
            if isinstance(parsed, Failure):
                print("Unhandled: ", mv)
        except Exception as e:
            logger.error("Exception parsing '%s' with agn parser", mv)

    dur = time.time() - t0
    print("time: {dur}, pieces/sec: {ps}, moves: {ln}",
          {"dur": dur, "ps": len(AN)/dur, "ln": len(AN)})


if __name__ == "__main__":
    main()
