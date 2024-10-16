import logging
import re

GLOBAL_LOGGING_LEVEL = logging.INFO
SIDES = ('W', 'B')
OPPO = {'W': 'B', 'B': 'W'}
PINS_RX = dict()
CHECKS_RX = dict()

for side in SIDES:
    PINS_RX[side] = re.compile(
        # beginning of string
        "^"
        # Current piece under contemplation
        f"(Q|R|B){side}"
        # zero or more empty squares before opponent piece
        f"(  )*"
        # ONE opponent piece
        f"((R|N|B|Q|P){OPPO[side]})"
        # zero or more empty squares after opponent piece
        f"(  )*"
        # opponent king
        f"K{OPPO[side]}"
        # end of string
        "$"
    )

    CHECKS_RX[side] = re.compile(
        # beginning of string
        "^"
        # Current piece under contemplation
        f"(Q|R|B){side}"
        # zero or more empty squares before opponent piece
        f"(  )*"
        # opponent king
        f"K{OPPO[side]}"
        # end of string
        "$"
    )
