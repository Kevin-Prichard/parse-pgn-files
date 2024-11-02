from copy import deepcopy
from unittest import TestCase

from deepdiff import DeepDiff

from parse_pgn import Ply, process_games_single, PGNStreamSlicer


# options = ["./parse_pgn.py", "-f", "../pgns.pgn", "-o", "/dev/stdout", "-l", "999", "-s", "-p", "1"]


class ProcessPGN(TestCase):
    def setUp(self):
        self.slicer = PGNStreamSlicer("tests/data/one.pgn")

    def test_simple_ply(self):
        r = process_games_single(pgn_parser=self.slicer, pgn_limit=1)
        g = r.ograph
        assert g.next[0] in g

    def test_process_games_single(self):
        res = process_games_single(pgn_parser=self.slicer, pgn_limit=1)
        ply = res.ograph
        cnt = 0
        while ply.next:
            ply = ply.next[0]
            cnt += 1
        # one.pgn contains 38.5 moves
        self.assertEqual(cnt, 38 * 2 + 1)


    def test_ply_merge(self):
        res = process_games_single(pgn_parser=self.slicer, pgn_limit=1)
        ply1 = res.ograph
        ply2 = deepcopy(ply1)
        ply1.merge(ply2)
        ply_count = 0
        while ply1.next:
            self.assertEqual(ply1.visits, 2)
            ply1 = ply1.next[0]
            ply_count += 1
        self.assertEqual(ply_count, 38 * 2 + 1)

    def test_ply_from_dict(self):
        res = process_games_single(pgn_parser=self.slicer, pgn_limit=1)
        ply1 = res.ograph
        ply1_raw = ply1.to_dict()
        ply2 = Ply.from_dict(ply1_raw)
        ply2_raw = ply2.to_dict()
        self.assertEqual(ply1, ply2)
        self.assertDictEqual(ply1_raw, ply2_raw)
        diffs = DeepDiff(ply1_raw, ply2_raw)
        self.assertDictEqual(diffs, {})

        # now, cause an inequality in the original ply graph and check for it
        # descend to the 10th ply...
        ply = ply1_raw
        for depth in range(10):
            ply = ply['next'][0]

        #  ...and change the number of visits
        ply['visits'] = 0

        # check 1
        self.assertNotEqual(ply1_raw, ply2_raw)

        # check 2
        diffs = DeepDiff(ply1_raw, ply2_raw)
        expected_diff_key = ("root['next'][0]['next'][0]['next'][0]['next'][0]"
                             "['next'][0]['next'][0]['next'][0]['next'][0]"
                             "['next'][0]['next'][0]['visits']")
        # check 2a
        self.assertEqual(list(diffs['values_changed'].keys())[0],
                         expected_diff_key)
        # check 2b
        self.assertEqual(diffs, {'values_changed': {
            expected_diff_key: {'new_value': 1, 'old_value': 0}}})

    def test_ply_visits_turtles_all_the_way_down(self):
        # Recurse the ply graph and check that the sum of visits at each level
        # is equal to the number of visits of the descendents

        def compare(ply, expected_visits, depth=0):
            visits = 0
            if not ply.next:
                return
            for p in ply.next:
                visits += p.visits
                compare(p, p.visits, depth + 1)
            self.assertEqual(visits, expected_visits)

        for pgn_file in ["tests/data/pgns25.pgn", "tests/data/pgns250.pgn"]:
            slicer = PGNStreamSlicer(pgn_file)
            res = process_games_single(pgn_parser=slicer, pgn_limit=25)
            ply = res.ograph
            compare(ply, ply.visits)
