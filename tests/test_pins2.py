from parse_pgn import Board, eval_king_state

from unittest import TestCase

import pudb


class TestPinning(TestCase):
    def setUp(self):
        self.b = Board(empty=True)

    def setup_board(self, pieces):
        for piece, square in pieces:
            self.b.add(piece, square)
        print(f"\n{self.b}\n")

    def test_bishop_pins_pawn_to_king_adjacent(self):
        self.setup_board([('BW', 'a1'), ('NB', 'b2'), ('KB', 'c3')])
        self.assertTrue(eval_king_state('B', 'a1', self.b, 'W'))
        self.assertIn(('NB', 'b2'), self.b._pinned)

    ###########################################################################
    # Bishop vs King: pinning to upper-right
    def test_bishop_pins_pawn_to_king_with_spaces_UR(self):
        self.setup_board([('BW', 'a1'), ('PB', 'c3'), ('KB', 'e5')])
        self.assertTrue(eval_king_state('B', 'a1', self.b, 'W'))
        self.assertIn(('PB', 'c3'), self.b._pinned)

    # Bishop vs King: pinning to lower-right
    def test_bishop_pins_pawn_to_king_with_spaces_LR(self):
        self.setup_board([('BW', 'c6'), ('PB', 'e4'), ('KB', 'g2')])
        self.assertTrue(eval_king_state('B', 'c6', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    # Bishop vs King: pinning to lower-left
    def test_bishop_pins_pawn_to_king_with_spaces_LL(self):
        self.setup_board([('BW', 'g6'), ('PB', 'e4'), ('KB', 'c2')])
        self.assertTrue(eval_king_state('B', 'g6', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    # Bishop vs King: pinning to upper-left
    def test_bishop_pins_pawn_to_king_with_spaces_UL(self):
        self.setup_board([('BW', 'f3'), ('PB', 'e4'), ('KB', 'd5')])
        self.assertTrue(eval_king_state('B', 'f3', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    ###########################################################################
    # Queen vs King: pinning to upper-right
    def test_queen_pins_pawn_to_king_with_spaces_UR(self):
        self.setup_board([('BW', 'a1'), ('PB', 'c3'), ('KB', 'e5')])
        self.assertTrue(eval_king_state('Q', 'a1', self.b, 'W'))
        self.assertIn(('PB', 'c3'), self.b._pinned)

    # Queen vs King: pinning to lower-right
    def test_queen_pins_pawn_to_king_with_spaces_LR(self):
        self.setup_board([('BW', 'c6'), ('PB', 'e4'), ('KB', 'g2')])
        self.assertTrue(eval_king_state('Q', 'c6', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    # Queen vs King: pinning to lower-left
    def test_queen_pins_pawn_to_king_with_spaces_LL(self):
        self.setup_board([('BW', 'g6'), ('PB', 'e4'), ('KB', 'c2')])
        self.assertTrue(eval_king_state('Q', 'g6', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    # Queen vs King: pinning to upper-left
    def test_queen_pins_pawn_to_king_with_spaces_UL(self):
        self.setup_board([('BW', 'f3'), ('PB', 'e4'), ('KB', 'd5')])
        self.assertTrue(eval_king_state('Q', 'f3', self.b, 'W'))
        self.assertIn(('PB', 'e4'), self.b._pinned)

    ###########################################################################
    # Queen vs King: pinning to right
    def test_queen_pins_pawn_to_king_to_right(self):
        self.setup_board([('QW', 'a1'), ('PB', 'c1'), ('KB', 'e1')])
        self.assertTrue(eval_king_state('Q', 'a1', self.b, 'W'))
        self.assertIn(('PB', 'c1'), self.b._pinned)

    # Queen vs King: pinning to below
    def test_queen_pins_pawn_to_king_to_bottom(self):
        self.setup_board([('QW', 'd4'), ('PB', 'd3'), ('KB', 'd1')])
        self.assertTrue(eval_king_state('Q', 'd4', self.b, 'W'))
        self.assertIn(('PB', 'd3'), self.b._pinned)

    # Queen vs King: pinning to left
    def test_queen_pins_pawn_to_king_to_left(self):
        self.setup_board([('QW', 'g6'), ('PB', 'c6'), ('KB', 'a6')])
        self.assertTrue(eval_king_state('Q', 'g6', self.b, 'W'))
        self.assertIn(('PB', 'c6'), self.b._pinned)

    # Queen vs King: pinning to upper-left
    def test_queen_pins_pawn_to_king_to_above(self):
        self.setup_board([('QW', 'e3'), ('PB', 'e5'), ('KB', 'e7')])
        self.assertTrue(eval_king_state('Q', 'e3', self.b, 'W'))
        self.assertIn(('PB', 'e5'), self.b._pinned)


    ###########################################################################
    # Queen vs King: pinning to upper-right
    def test_queen_pins_knight_to_king_with_spaces_UR(self):
        self.setup_board([('BW', 'a1'), ('NB', 'c3'), ('KB', 'e5')])
        self.assertTrue(eval_king_state('Q', 'a1', self.b, 'W'))
        self.assertIn(('NB', 'c3'), self.b._pinned)

    # Queen vs King: pinning to lower-right
    def test_queen_pins_knight_to_king_with_spaces_LR(self):
        self.setup_board([('BW', 'c6'), ('NB', 'e4'), ('KB', 'g2')])
        self.assertTrue(eval_king_state('Q', 'c6', self.b, 'W'))
        self.assertIn(('NB', 'e4'), self.b._pinned)

    # Queen vs King: pinning to lower-left
    def test_queen_pins_knight_to_king_with_spaces_LL(self):
        self.setup_board([('BW', 'g6'), ('NB', 'e4'), ('KB', 'c2')])
        self.assertTrue(eval_king_state('Q', 'g6', self.b, 'W'))
        self.assertIn(('NB', 'e4'), self.b._pinned)

    # Queen vs King: pinning to upper-left
    def test_queen_pins_knight_to_king_with_spaces_UL(self):
        self.setup_board([('BW', 'f3'), ('NB', 'e4'), ('KB', 'd5')])
        self.assertTrue(eval_king_state('Q', 'f3', self.b, 'W'))
        self.assertIn(('NB', 'e4'), self.b._pinned)

    ###########################################################################
    # Queen vs King: pinning to right
    def test_queen_pins_knight_to_king_to_right(self):
        self.setup_board([('QW', 'a1'), ('NB', 'c1'), ('KB', 'e1')])
        self.assertTrue(eval_king_state('Q', 'a1', self.b, 'W'))
        self.assertIn(('NB', 'c1'), self.b._pinned)

    # Queen vs King: pinning to below
    def test_queen_pins_knight_to_king_to_bottom(self):
        self.setup_board([('QW', 'd4'), ('NB', 'd3'), ('KB', 'd1')])
        self.assertTrue(eval_king_state('Q', 'd4', self.b, 'W'))
        self.assertIn(('NB', 'd3'), self.b._pinned)

    # Queen vs King: pinning to left
    def test_queen_pins_knight_to_king_to_left(self):
        self.setup_board([('QW', 'g6'), ('NB', 'c6'), ('KB', 'a6')])
        self.assertTrue(eval_king_state('Q', 'g6', self.b, 'W'))
        self.assertIn(('NB', 'c6'), self.b._pinned)

    # Queen vs King: pinning to upper-left
    def test_queen_pins_knight_to_king_to_above(self):
        self.setup_board([('QW', 'e3'), ('NB', 'e5'), ('KB', 'e7')])
        self.assertTrue(eval_king_state('Q', 'e3', self.b, 'W'))
        self.assertIn(('NB', 'e5'), self.b._pinned)

    ###########################################################################
    # Rook vs King: pinning to right
    def test_rook_pins_pawn_to_king_to_right(self):
        self.setup_board([('RW', 'a1'), ('PB', 'c1'), ('KB', 'e1')])
        self.assertTrue(eval_king_state('R', 'a1', self.b, 'W'))
        self.assertIn(('PB', 'c1'), self.b._pinned)

    # Rook vs King: pinning to below
    def test_rook_pins_pawn_to_king_to_bottom(self):
        self.setup_board([('RW', 'd4'), ('PB', 'd3'), ('KB', 'd1')])
        self.assertTrue(eval_king_state('R', 'd4', self.b, 'W'))
        self.assertIn(('PB', 'd3'), self.b._pinned)

    # Rook vs King: pinning to left
    def test_rook_pins_pawn_to_king_to_left(self):
        self.setup_board([('RW', 'g6'), ('PB', 'c6'), ('KB', 'a6')])
        self.assertTrue(eval_king_state('R', 'g6', self.b, 'W'))
        self.assertIn(('PB', 'c6'), self.b._pinned)

    # Rook vs King: pinning to upper-left
    def test_rook_pins_pawn_to_king_to_above(self):
        self.setup_board([('RW', 'e3'), ('PB', 'e5'), ('KB', 'e7')])
        self.assertTrue(eval_king_state('R', 'e3', self.b, 'W'))
        self.assertIn(('PB', 'e5'), self.b._pinned)
