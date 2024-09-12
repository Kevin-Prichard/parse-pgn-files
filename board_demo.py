#!/usr/bin/env python3

from datetime import datetime
import queue


import PySimpleGUI as sg
import os
import sys
import threading
from pathlib import Path, PurePath  # Python 3.4 and up
import queue
import copy
import time
from datetime import datetime
import json
import pyperclip
import chess
import chess.pgn
import chess.engine
import chess.polyglot
import logging
import platform as sys_plat


log_format = '%(asctime)s :: %(funcName)s :: line: %(lineno)d :: %(levelname)s :: %(message)s'
logging.basicConfig(
    filename='pecg_log.txt',
    filemode='w',
    level=logging.DEBUG,
    format=log_format
)


APP_NAME = 'Python Easy Chess GUI'
APP_VERSION = 'v1.19.0'
BOX_TITLE = f'{APP_NAME} {APP_VERSION}'


platform = sys.platform
sys_os = sys_plat.system()


ico_path = {
    'win32': {'pecg': 'Icon/pecg.ico', 'enemy': 'Icon/enemy.ico', 'adviser': 'Icon/adviser.ico'},
    'linux': {'pecg': 'Icon/pecg.png', 'enemy': 'Icon/enemy.png', 'adviser': 'Icon/adviser.png'},
    'darwin': {'pecg': 'Icon/pecg.png', 'enemy': 'Icon/enemy.png', 'adviser': 'Icon/adviser.png'}
}


MIN_DEPTH = 1
MAX_DEPTH = 1000
MANAGED_UCI_OPTIONS = ['ponder', 'uci_chess960', 'multipv', 'uci_analysemode', 'ownbook']
GUI_THEME = [
    'Green', 'GreenTan', 'LightGreen', 'BluePurple', 'Purple', 'BlueMono', 'GreenMono', 'BrownBlue',
    'BrightColors', 'NeutralBlue', 'Kayak', 'SandyBeach', 'TealMono', 'Topanga', 'Dark', 'Black', 'DarkAmber'
]

IMAGE_PATH = 'Images/60'  # path to the chess pieces


BLANK = 0  # piece names
PAWNB = 1
KNIGHTB = 2
BISHOPB = 3
ROOKB = 4
KINGB = 5
QUEENB = 6
PAWNW = 7
KNIGHTW = 8
BISHOPW = 9
ROOKW = 10
KINGW = 11
QUEENW = 12


# Absolute rank based on real chess board, white at bottom, black at the top.
# This is also the rank mapping used by python-chess modules.
RANK_8 = 7
RANK_7 = 6
RANK_6 = 5
RANK_5 = 4
RANK_4 = 3
RANK_3 = 2
RANK_2 = 1
RANK_1 = 0


initial_board = [[ROOKB, KNIGHTB, BISHOPB, QUEENB, KINGB, BISHOPB, KNIGHTB, ROOKB],
                 [PAWNB, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [PAWNW, ] * 8,
                 [ROOKW, KNIGHTW, BISHOPW, QUEENW, KINGW, BISHOPW, KNIGHTW, ROOKW]]


white_init_promote_board = [[QUEENW, ROOKW, BISHOPW, KNIGHTW]]

black_init_promote_board = [[QUEENB, ROOKB, BISHOPB, KNIGHTB]]


HELP_MSG = """The GUI has 2 modes, Play and Neutral. After startup
you are in Neutral mode. You can go to mode Play through Mode menu.
"""

# Images/60
blank = os.path.join(IMAGE_PATH, 'blank.png')
bishopB = os.path.join(IMAGE_PATH, 'bB.png')
bishopW = os.path.join(IMAGE_PATH, 'wB.png')
pawnB = os.path.join(IMAGE_PATH, 'bP.png')
pawnW = os.path.join(IMAGE_PATH, 'wP.png')
knightB = os.path.join(IMAGE_PATH, 'bN.png')
knightW = os.path.join(IMAGE_PATH, 'wN.png')
rookB = os.path.join(IMAGE_PATH, 'bR.png')
rookW = os.path.join(IMAGE_PATH, 'wR.png')
queenB = os.path.join(IMAGE_PATH, 'bQ.png')
queenW = os.path.join(IMAGE_PATH, 'wQ.png')
kingB = os.path.join(IMAGE_PATH, 'bK.png')
kingW = os.path.join(IMAGE_PATH, 'wK.png')


images = {
    BISHOPB: bishopB, BISHOPW: bishopW, PAWNB: pawnB, PAWNW: pawnW,
    KNIGHTB: knightB, KNIGHTW: knightW, ROOKB: rookB, ROOKW: rookW,
    KINGB: kingB, KINGW: kingW, QUEENB: queenB, QUEENW: queenW, BLANK: blank
}


# Promote piece from psg (pysimplegui) to pyc (python-chess)
promote_psg_to_pyc = {
    KNIGHTB: chess.KNIGHT, BISHOPB: chess.BISHOP,
    ROOKB: chess.ROOK, QUEENB: chess.QUEEN,
    KNIGHTW: chess.KNIGHT, BISHOPW: chess.BISHOP,
    ROOKW: chess.ROOK, QUEENW: chess.QUEEN
}


INIT_PGN_TAG = {
    'Event': 'Human vs computer',
    'White': 'Human',
    'Black': 'Computer'
}


# (1) Mode: Neutral
"""
menu_def_neutral = [
        ['&Mode', ['Play']],
        ['Boar&d', ['Flip', 'Color', ['Brown::board_color_k',
                                      'Blue::board_color_k',
                                      'Green::board_color_k',
                                      'Gray::board_color_k'],
                    'Theme', GUI_THEME]],
        ['&Engine', ['Set Engine Adviser', 'Set Engine Opponent', 'Set Depth',
                     'Manage', ['Install', 'Edit', 'Delete']]],
        ['&Time', ['User::tc_k', 'Engine::tc_k']],
        ['&Book', ['Set Book::book_set_k']],
        ['&User', ['Set Name::user_name_k']],
        ['Tools', ['PGN', ['Delete Player::delete_player_k']]],
        ['&Settings', ['Game::settings_game_k']],
        ['&Help', ['GUI']],
]

# (2) Mode: Play, info: hide
menu_def_play = [
        ['&Mode', ['Neutral']],
        ['&Game', ['&New::new_game_k',
                   'Save to My Games::save_game_k',
                   'Save to White Repertoire',
                   'Save to Black Repertoire',
                   'Resign::resign_game_k',
                   'User Wins::user_wins_k',
                   'User Draws::user_draws_k']],
        ['FEN', ['Paste']],
        ['&Engine', ['Go', 'Move Now']],
        ['&Help', ['GUI']],
]
"""


class EasyChessGui:
    queue = queue.Queue()
    is_user_white = True  # White is at the bottom in board layout

    def __init__(self, theme):
        self.theme = theme
        self.username = 'Human'
        self.gui_theme = 'Reddit'

        self.init_game()
        self.fen = None
        self.psg_board = None
        self.menu_elem = None
        self.engine_id_name_list = []
        self.engine_file_list = []
        self.username = 'Human'

        self.human_base_time_ms = 5 * 60 * 1000  # 5 minutes
        self.human_inc_time_ms = 10 * 1000  # 10 seconds
        self.human_period_moves = 0
        self.human_tc_type = 'fischer'

        self.engine_base_time_ms = 3 * 60 * 1000  # 5 minutes
        self.engine_inc_time_ms = 2 * 1000  # 10 seconds
        self.engine_period_moves = 0
        self.engine_tc_type = 'fischer'

        # Default board color is brown
        self.sq_light_color = '#F0D9B5'
        self.sq_dark_color = '#B58863'

        # Move highlight, for brown board
        self.move_sq_light_color = '#E8E18E'
        self.move_sq_dark_color = '#B8AF4E'

        self.gui_theme = 'Reddit'

        self.is_save_time_left = False
        self.is_save_user_comment = True

    def update_game(self, mc: int, user_move: str, user_comment: str=None):
        """Saves moves in the game.

        Args:
          mc: move count
          user_move: user's move
          time_left: time left
          user_comment: Can be a 'book' from the engine
        """
        # Save user comment
        if self.is_save_user_comment:
            # If comment is empty
            if not (user_comment and user_comment.strip()):
                if mc == 1:
                    self.node = self.game.add_variation(user_move)
                else:
                    self.node = self.node.add_variation(user_move)
            else:
                if mc == 1:
                    self.node = self.game.add_variation(user_move)
                else:
                    self.node = self.node.add_variation(user_move)
        # Do not save user comment
        else:
            if mc == 1:
                self.node = self.game.add_variation(user_move)
            else:
                self.node = self.node.add_variation(user_move)

    def create_new_window(self, window, flip=False):
        """Hide current window and creates a new window."""
        loc = window.CurrentLocation()
        window.Hide()
        if flip:
            self.is_user_white = not self.is_user_white

        layout = self.build_main_layout(self.is_user_white)

        w = sg.Window(
            'yah',
            layout,
            default_button_element_size=(12, 1),
            auto_size_buttons=False,
            location=(loc[0], loc[1]),
            icon=ico_path[platform]['pecg']
        )

        # Initialize White and black boxes
        while True:
            button, value = w.Read(timeout=50)
            # self.update_labels_and_game_tags(w, human=self.username)
            break

        window.Close()
        return w

    def update_text_box(self, window, msg, is_hide):
        """ Update text elements """
        best_move = None
        msg_str = str(msg)

        if 'bestmove ' not in msg_str:
            if 'info_all' in msg_str:
                info_all = ' '.join(msg_str.split()[0:-1]).strip()
                msg_line = '{}\n'.format(info_all)
                window.find_element('search_info_all_k').Update(
                        '' if is_hide else msg_line)
        else:
            # Best move can be None because engine dies
            try:
                best_move = chess.Move.from_uci(msg.split()[1])
            except Exception:
                logging.exception(f'Engine sent {best_move}')
                sg.Popup(
                    f'Engine error, it sent a {best_move} bestmove.\n \
                    Back to Neutral mode, it is better to change engine {self.opp_id_name}.',
                    icon=ico_path[platform]['pecg'],
                    title=BOX_TITLE
                )

        return best_move

    def get_tag_date(self):
        """ Return date in pgn tag date format """
        return datetime.today().strftime('%Y.%m.%d')

    def init_game(self):
        """ Initialize game with initial pgn tag values """
        self.game = chess.pgn.Game()
        self.node = None
        self.game.headers['Event'] = INIT_PGN_TAG['Event']
        self.game.headers['Date'] = self.get_tag_date()
        self.game.headers['White'] = INIT_PGN_TAG['White']
        self.game.headers['Black'] = INIT_PGN_TAG['Black']

    def set_new_game(self):
        """ Initialize new game but save old pgn tag values"""
        old_event = self.game.headers['Event']
        old_white = self.game.headers['White']
        old_black = self.game.headers['Black']

        # Define a game object for saving game in pgn format
        self.game = chess.pgn.Game()

        self.game.headers['Event'] = old_event
        self.game.headers['Date'] = self.get_tag_date()
        self.game.headers['White'] = old_white
        self.game.headers['Black'] = old_black

    def clear_elements(self, window):
        """ Clear movelist, score, pv, time, depth and nps boxes """
        window.find_element('search_info_all_k').Update('')
        window.find_element('_movelist_').Update(disabled=False)
        window.find_element('_movelist_').Update('', disabled=True)
        window.find_element('polyglot_book1_k').Update('')
        window.find_element('polyglot_book2_k').Update('')
        window.find_element('advise_info_k').Update('')
        window.find_element('comment_k').Update('')
        window.Element('w_base_time_k').Update('')
        window.Element('b_base_time_k').Update('')
        window.Element('w_elapse_k').Update('')
        window.Element('b_elapse_k').Update('')

    # def update_labels_and_game_tags(self, window, human='Human'):
    #     """ Update player names """
    #     engine_id = "Opponent"
    #     if self.is_user_white:
    #         window.find_element('_White_').Update(human)
    #         window.find_element('_Black_').Update(engine_id)
    #         self.game.headers['White'] = human
    #         self.game.headers['Black'] = engine_id
    #     else:
    #         window.find_element('_White_').Update(engine_id)
    #         window.find_element('_Black_').Update(human)
    #         self.game.headers['White'] = engine_id
    #         self.game.headers['Black'] = human
    #
    # def get_fen(self):
    #     """ Get fen from clipboard """
    #     self.fen = pyperclip.paste()
    #
    #     # Remove empty char at the end of FEN
    #     if self.fen.endswith(' '):
    #         self.fen = self.fen[:-1]
    #
    # def fen_to_psg_board(self, window):
    #     """ Update psg_board based on FEN """
    #     psgboard = []
    #
    #     # Get piece locations only to build psg board
    #     pc_locations = self.fen.split()[0]
    #
    #     board = chess.BaseBoard(pc_locations)
    #     old_r = None
    #
    #     for s in chess.SQUARES:
    #         r = chess.square_rank(s)
    #
    #         if old_r is None:
    #             piece_r = []
    #         elif old_r != r:
    #             psgboard.append(piece_r)
    #             piece_r = []
    #         elif s == 63:
    #             psgboard.append(piece_r)
    #
    #         try:
    #             pc = board.piece_at(s ^ 56)
    #         except Exception:
    #             pc = None
    #             logging.exception('Failed to get piece.')
    #
    #         if pc is not None:
    #             pt = pc.piece_type
    #             c = pc.color
    #             if c:
    #                 if pt == chess.PAWN:
    #                     piece_r.append(PAWNW)
    #                 elif pt == chess.KNIGHT:
    #                     piece_r.append(KNIGHTW)
    #                 elif pt == chess.BISHOP:
    #                     piece_r.append(BISHOPW)
    #                 elif pt == chess.ROOK:
    #                     piece_r.append(ROOKW)
    #                 elif pt == chess.QUEEN:
    #                     piece_r.append(QUEENW)
    #                 elif pt == chess.KING:
    #                     piece_r.append(KINGW)
    #             else:
    #                 if pt == chess.PAWN:
    #                     piece_r.append(PAWNB)
    #                 elif pt == chess.KNIGHT:
    #                     piece_r.append(KNIGHTB)
    #                 elif pt == chess.BISHOP:
    #                     piece_r.append(BISHOPB)
    #                 elif pt == chess.ROOK:
    #                     piece_r.append(ROOKB)
    #                 elif pt == chess.QUEEN:
    #                     piece_r.append(QUEENB)
    #                 elif pt == chess.KING:
    #                     piece_r.append(KINGB)
    #
    #         # Else if pc is None or square is empty
    #         else:
    #             piece_r.append(BLANK)
    #
    #         old_r = r
    #
    #     self.psg_board = psgboard
    #     self.redraw_board(window)

    def change_square_color(self, window, row, col):
        """
        Change the color of a square based on square row and col.
        """
        btn_sq = window.find_element(key=(row, col))
        is_dark_square = True if (row + col) % 2 else False
        bd_sq_color = self.move_sq_dark_color if is_dark_square else self.move_sq_light_color
        btn_sq.Update(button_color=('white', bd_sq_color))

    def relative_row(self, s, stm):
        """
        The board can be viewed, as white at the bottom and black at the
        top. If stm is white the row 0 is at the bottom. If stm is black
        row 0 is at the top.
        :param s: square
        :param stm: side to move
        :return: relative row
        """
        return 7 - self.get_row(s) if stm else self.get_row(s)

    def get_row(self, s):
        """
        This row is based on PySimpleGUI square mapping that is 0 at the
        top and 7 at the bottom.
        In contrast Python-chess square mapping is 0 at the bottom and 7
        at the top. chess.square_rank() is a method from Python-chess that
        returns row given square s.

        :param s: square
        :return: row
        """
        return 7 - chess.square_rank(s)

    def get_col(self, s):
        """ Returns col given square s """
        return chess.square_file(s)

    def redraw_board(self, window):
        """
        Redraw board at start and afte a move.

        :param window:
        :return:
        """
        for i in range(8):
            for j in range(8):
                color = self.sq_dark_color if (i + j) % 2 else \
                        self.sq_light_color
                piece_image = images[self.psg_board[i][j]]
                elem = window.find_element(key=(i, j))
                elem.Update(button_color=('white', color),
                            image_filename=piece_image, )

    def render_square(self, image, key, location):
        """ Returns an RButton (Read Button) with image image """
        if (location[0] + location[1]) % 2:
            color = self.sq_dark_color  # Dark square
        else:
            color = self.sq_light_color
        return sg.RButton('', image_filename=image, size=(1, 1),
                          border_width=0, button_color=('white', color),
                          pad=(0, 0), key=key)

    def select_promotion_piece(self, stm):
        """
        Allow user to select a piece type to promote to.

        :param stm: side to move
        :return: promoted piece, i.e QUEENW, QUEENB ...
        """
        piece = None
        board_layout, row = [], []

        psg_promote_board = copy.deepcopy(white_init_promote_board) if stm else copy.deepcopy(black_init_promote_board)

        # Loop through board and create buttons with images.
        for i in range(1):
            for j in range(4):
                piece_image = images[psg_promote_board[i][j]]
                row.append(self.render_square(piece_image, key=(i, j),
                                              location=(i, j)))

            board_layout.append(row)

        promo_window = sg.Window('{} {}'.format(APP_NAME, APP_VERSION),
                                 board_layout,
                                 default_button_element_size=(12, 1),
                                 auto_size_buttons=False,
                                 icon=ico_path[platform]['pecg'])

        while True:
            button, value = promo_window.Read(timeout=0)
            if button is None:
                break
            if type(button) is tuple:
                move_from = button
                fr_row, fr_col = move_from
                piece = psg_promote_board[fr_row][fr_col]
                logging.info(f'promote piece: {piece}')
                break

        promo_window.Close()

        return piece

    def update_rook(self, window, move):
        """
        Update rook location for castle move.

        :param window:
        :param move: uci move format
        :return:
        """
        if move == 'e1g1':
            fr = chess.H1
            to = chess.F1
            pc = ROOKW
        elif move == 'e1c1':
            fr = chess.A1
            to = chess.D1
            pc = ROOKW
        elif move == 'e8g8':
            fr = chess.H8
            to = chess.F8
            pc = ROOKB
        elif move == 'e8c8':
            fr = chess.A8
            to = chess.D8
            pc = ROOKB

        self.psg_board[self.get_row(fr)][self.get_col(fr)] = BLANK
        self.psg_board[self.get_row(to)][self.get_col(to)] = pc
        self.redraw_board(window)

    def update_ep(self, window, move, stm):
        """
        Update board for e.p move.

        :param window:
        :param move: python-chess format
        :param stm: side to move
        :return:
        """
        to = move.to_square
        if stm:
            capture_sq = to - 8
        else:
            capture_sq = to + 8

        self.psg_board[self.get_row(capture_sq)][self.get_col(capture_sq)] = BLANK
        self.redraw_board(window)

    def create_board(self, is_user_white=True):
        """
        Returns board layout based on color of user. If user is white,
        the white pieces will be at the bottom, otherwise at the top.

        :param is_user_white: user has handling the white pieces
        :return: board layout
        """
        file_char_name = 'abcdefgh'
        self.psg_board = copy.deepcopy(initial_board)

        board_layout = []

        if is_user_white:
            # Save the board with black at the top.
            start = 0
            end = 8
            step = 1
        else:
            start = 7
            end = -1
            step = -1
            file_char_name = file_char_name[::-1]

        # Loop through the board and create buttons with images
        for i in range(start, end, step):
            # Row numbers at left of board is blank
            row = []
            for j in range(start, end, step):
                piece_image = images[self.psg_board[i][j]]
                row.append(self.render_square(piece_image, key=(i, j), location=(i, j)))
            board_layout.append(row)

        return board_layout

    def build_main_layout(self, is_user_white=True):
        """
        Creates all elements for the GUI, icluding the board layout.

        :param is_user_white: if user is white, the white pieces are
        oriented such that the white pieces are at the bottom.
        :return: GUI layout
        """
        sg.ChangeLookAndFeel(self.gui_theme)
        sg.SetOptions(margins=(0, 3), border_width=1)

        # Define board
        board_layout = self.create_board(is_user_white)

        board_controls = []
        #     [sg.Text('Mode     Neutral', size=(36, 1), font=('Consolas', 10), key='_gamestatus_')],
        #     [sg.Text('White', size=(7, 1), font=('Consolas', 10)),
        #      sg.Text('Human', font=('Consolas', 10), key='_White_',
        #              size=(24, 1), relief='sunken'),
        #      sg.Text('', font=('Consolas', 10), key='w_base_time_k',
        #              size=(11, 1), relief='sunken'),
        #      sg.Text('', font=('Consolas', 10), key='w_elapse_k', size=(7, 1),
        #              relief='sunken')
        #      ],
        #     [sg.Text('Black', size=(7, 1), font=('Consolas', 10)),
        #      sg.Text('Computer', font=('Consolas', 10), key='_Black_',
        #              size=(24, 1), relief='sunken'),
        #      sg.Text('', font=('Consolas', 10), key='b_base_time_k',
        #              size=(11, 1), relief='sunken'),
        #      sg.Text('', font=('Consolas', 10), key='b_elapse_k', size=(7, 1),
        #              relief='sunken')
        #      ],
        #     [sg.Text('Adviser', size=(7, 1), font=('Consolas', 10), key='adviser_k',
        #              right_click_menu=[
        #                 'Right',
        #                 ['Start::right_adviser_k', 'Stop::right_adviser_k']
        #             ]),
        #      sg.Text('', font=('Consolas', 10), key='advise_info_k', relief='sunken',
        #              size=(46, 1))],
        #
        #     [sg.Text('Move list', size=(16, 1), font=('Consolas', 10))],
        #     [sg.Multiline('', do_not_clear=True, autoscroll=True, size=(52, 8),
        #                   font=('Consolas', 10), key='_movelist_', disabled=True)],
        #
        #     [sg.Text('Comment', size=(7, 1), font=('Consolas', 10))],
        #     [sg.Multiline('', do_not_clear=True, autoscroll=True, size=(52, 3),
        #                   font=('Consolas', 10), key='comment_k')],
        #
        #     [sg.Text('BOOK 1, Comp games', size=(26, 1),
        #              font=('Consolas', 10),
        #              right_click_menu=['Right', ['Show::right_book1_k', 'Hide::right_book1_k']]),
        #      sg.Text('BOOK 2, Human games',
        #              font=('Consolas', 10),
        #              right_click_menu=['Right', ['Show::right_book2_k', 'Hide::right_book2_k']])],
        #     [sg.Multiline('', do_not_clear=True, autoscroll=False, size=(23, 4),
        #                   font=('Consolas', 10), key='polyglot_book1_k', disabled=True),
        #      sg.Multiline('', do_not_clear=True, autoscroll=False, size=(25, 4),
        #                   font=('Consolas', 10), key='polyglot_book2_k', disabled=True)],
        #     [sg.Text('Opponent Search Info', font=('Consolas', 10), size=(30, 1),
        #              right_click_menu=['Right',
        #                                ['Show::right_search_info_k', 'Hide::right_search_info_k']])],
        #     [sg.Text('', key='search_info_all_k', size=(55, 1),
        #              font=('Consolas', 10), relief='sunken')],
        # ]

        board_tab = [[sg.Column(board_layout)]]

        # self.menu_elem = sg.Menu(menu_def_neutral, tearoff=False)

        # White board layout, mode: Neutral
        layout = [
                # [self.menu_elem],
                [sg.Column(board_tab), sg.Column(board_controls)]
        ]

        return layout

    def main_loop_init(self):
        """
        Build GUI, read user and engine config files and take user inputs.

        :return:
        """
        engine_id_name = None
        layout = self.build_main_layout(True)

        # Use white layout as default window
        window = sg.Window('{} {}'.format(APP_NAME, APP_VERSION),
                           layout, default_button_element_size=(12, 1),
                           auto_size_buttons=False,
                           icon=ico_path[platform]['pecg'])

        self.init_game()

        # Initialize White and black boxes
        while True:
            button, value = window.Read(timeout=50)
            # self.update_labels_and_game_tags(window, human=self.username)
            break
        return window

    def main_loop(self, window):

        # Mode: Neutral, main loop starts here
        while True:
            # import pudb; pu.db
            button, value = window.Read(timeout=50)

            # Mode: Neutral, Change theme
            if button in GUI_THEME:
                self.gui_theme = button
                window = self.create_new_window(window)
                continue

            # Mode: Neutral, Change board to gray
            if button == 'Gray::board_color_k':
                self.sq_light_color = '#D8D8D8'
                self.sq_dark_color = '#808080'
                self.move_sq_light_color = '#e0e0ad'
                self.move_sq_dark_color = '#999966'
                self.redraw_board(window)
                window = self.create_new_window(window)
                continue

            # Mode: Neutral, Change board to green
            if button == 'Green::board_color_k':
                self.sq_light_color = '#daf1e3'
                self.sq_dark_color = '#3a7859'
                self.move_sq_light_color = '#bae58f'
                self.move_sq_dark_color = '#6fbc55'
                self.redraw_board(window)
                window = self.create_new_window(window)
                continue

            # Mode: Neutral, Change board to blue
            if button == 'Blue::board_color_k':
                self.sq_light_color = '#b9d6e8'
                self.sq_dark_color = '#4790c0'
                self.move_sq_light_color = '#d2e4ba'
                self.move_sq_dark_color = '#91bc9c'
                self.redraw_board(window)
                window = self.create_new_window(window)
                continue

            # Mode: Neutral, Change board to brown, default
            if button == 'Brown::board_color_k':
                self.sq_light_color = '#F0D9B5'
                self.sq_dark_color = '#B58863'
                self.move_sq_light_color = '#E8E18E'
                self.move_sq_dark_color = '#B8AF4E'
                self.redraw_board(window)
                window = self.create_new_window(window)
                continue

            # Mode: Neutral
            if button == 'Flip':
                window.find_element('_gamestatus_').Update('Mode     Neutral')
                self.clear_elements(window)
                window = self.create_new_window(window, True)
                continue

            # Mode: Neutral
            if button == 'GUI':
                sg.PopupScrolled(HELP_MSG, title='Help/GUI')
                continue

        window.Close()


def prepare_board():
    game = EasyChessGui(theme='Reddit')
    window = game.main_loop_init()
    return game, window
    # game.main_loop(w)


if __name__ == '__main__':
    prepare_board()
