import logging

from parsita import lit, opt, reg, rep, repsep

"""
PGN grammar-parser

Includes support for-
 - Annotations
 - Extended move syntax (LiChess's {comment} syntax)
 - Move decomposition
"""

logger = logging.getLogger(__name__)


def formatannotations(annotations):
    return {ant[0]: ant[1] for ant in annotations}


def formatgame(game):
    return {
        'moves': game[0],
        'outcome': game[1]
    }


def formatentry(entry):
    return {'annotations': entry[0], 'game': entry[1]}


def handle_optional(optionalmove):
    if len(optionalmove) > 0:
        return optionalmove[0]
    else:
        return None


def handle_move(move):
    """
    Normal move:
    [1, [['e4', []], ['{ [%clk 0:03:00] }']], [['c5', []], ['{ [%clk 0:03:00] }']]]

    Split move:
    [1, [['d4', []], ['{ [%eval 0.08] [%clk 0:05:00] }']], [1, [['d5', []], ['{ [%eval 0.07] [%clk 0:05:00] }']]]]

    # White move number @ [0]
    [1,
        # White algebraic move @ [1][0][0]
        [['d4', []],
                     # White comment @ [1][1][0][0]
                     ['{ [%eval 0.08] [%clk 0:05:00] }']],
    # Black move number @ [2][0]
        [1,
            # Black algebraic move @ [2][1][0][0]
            [['d5', []],
                         # Black comment @ [2][1][1][0]
                         ['{ [%eval 0.07] [%clk 0:05:00] }']]]]
    """
    # TODO: param 'move' is ugly and convoluted, I would like to restructure
    #  it, tho generally that means functions,
    #  which tend to decrease performance

    res = None
    try:
        # Base rsponse w/White's move
        res = {
            'num': move[0],
            'white': (move[1][0][0],
                      move[1][0][1][0] if move[1][0][1] else None,
                      move[1][1][0] if move[1][1] else None),
            'black': None,
        }

        # Handle split move for Black
        if move[2] and isinstance(move[2][0], int) and move[2][1]:
            res['black'] = (move[2][1][0][0],
                            move[2][1][0][1][0] if move[2][1][0][1] else None,
                            move[2][1][1][0] if move[2][1][1] else None)

        # Regular move for Black
        elif move[2] and isinstance(move[2][0], list):
            res['black'] = (move[2][0][0],
                            move[2][0][1][0] if move[2][0][1] else None,
                            move[2][1][0] if move[2][1] else None)

    except Exception as e:
        logger.error(f"Error handling move: {move}")
    return res


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
    r'[a-h1-8NBRQKx=]+')  # Matches more than just chess moves
longcastle = reg(
    r'O-O-O')  # match first to avoid castle matching spuriously
castle = reg(r'O-O')
nullmove = lit('--')  # Illegal move rarely used in annotations
unfinished = lit('*')  # Game unfinished

move_annotation = rep(
    lit('+') |   # check
    lit('#') |   # checkmate
    lit('!!') |  # brilliant—and usually surprising—move
    lit('!?') |  # interesting move that may not be the best
    lit('!') |   # very good move
    lit('?!') |  # dubious move that is not easily refutable
    lit('??') |  # blunder
    lit('?') |   # bad move; a mistake
    lit('⌓') |   # better move than the one played
    lit('□') |   # forced move; the only reasonable move, or the only move available
    lit('TN') | lit('NA')   # or NA
)

# LiChess annotations
move_comment = reg(r'\{[^}]*\}')

# Possible move types
move = (regularmove | longcastle | castle | nullmove) & opt(move_annotation)

# Build up the game
move_number = (reg(r'[0-9]+') << '.' << whitespace) > int
move_number_ellipsis = (reg(r'[0-9]+') << '...' << whitespace) > int

standard_turn = move_number & (
            move << whitespace & (opt(move_comment << whitespace))) & (
                opt(move << whitespace & (
                   opt(move_comment << whitespace))) > handle_optional)
split_turn = move_number & (
            move << whitespace & (opt(move_comment << whitespace))) & (
            move_number_ellipsis & (
                opt(move << whitespace & (
                   opt(move_comment << whitespace))) > handle_optional))
turn = (standard_turn | split_turn) > handle_move

draw = lit('1/2-1/2')
white = lit('1-0')
black = lit('0-1')
outcome = draw | white | black | unfinished

game = (rep(turn) & outcome) > formatgame

# A PGN entry is annotations and the game
entry = ((annotations << rep(whitespace)) & (
        game << rep(whitespace))) > formatentry

# A file is repeated entries
pgn_file = rep(entry)
