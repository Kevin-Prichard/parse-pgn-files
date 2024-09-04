#!/usr/bin/env python3
#cython: language_level=3

import argparse
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
    b'': PGNParts.ANNOTATION,
    b'[': PGNParts.ANNOTATION,
    b'1': PGNParts.MOVES,
    b'\n': PGNParts.NEWLINE,
}


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()


def get_args(argv: List[str]) -> Tuple[argparse.Namespace, ArgumentParser]:
    parser = ArgumentParser(
        prog='./parse_pgn.py',
        description='Parse PGN file(s) and stream(s): '
                    'parse PGNs and pass the resulting '
                    'parse tree to a function.')
    parser.add_argument('--file', '-f', dest='pgn_path',
                        type=str, action='store', default='-')
    parser.add_argument('--output', '-o', dest='out_path',
                        type=str, action='store', default='/dev/stdout')
    parser.add_argument('--processes', '-p', dest='process_count',
                        type=int, action='store', default=mp.cpu_count(),
                        help='Number of processes to use; 1=single tasking')
    parser.add_argument('--limit', '-l', dest='pgn_limit',
                        type=int, action='store', default=0)
    parser.add_argument('--queue_limit', '-q', dest='queue_limit',
                        type=int, action='store', default=250)
    return parser.parse_args(argv), parser


class PGNStreamSlicer:
    def __init__(self, pgn_path: str, parse_cb: Callable[[Dict], None] = None):
        self.pgn_path = pgn_path
        self.pgn_file = open(self.pgn_path, 'rb')  # rb: PGNs are UTF-8 encoded, e.g. "Réti Opening"
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
            if ( this_type == PGNParts.ANNOTATION and
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


def handle_pgn(pgn, results: dict):
    result = pgn_file.parse(pgn)
    if isinstance(result, Success):
        result = result.unwrap()[0]
        results['gcount'] += 1

        for num, a_move in enumerate(result['game']['moves']):
            try:
                if a_move:
                    results['mcount'] += 1
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


def main(argv):
    args, parser = get_args(argv)
    parser = PGNStreamSlicer(pgn_path=args.pgn_path)
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
        }))

    return 0.0


if __name__ == "__main__":
    main(sys.argv[1:])
