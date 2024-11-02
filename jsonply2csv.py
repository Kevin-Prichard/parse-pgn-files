#!/usr/bin/env python3

import sys
import json
from typing import List


NativePly = List[str | List]
CSVRecord = List[str]

row_no = 0


def compact3ply2json(ply_root: NativePly, file_nr: str) -> List[str]:
    """
    Convert the Ply class graph into CSV records.
    Each record captures one entire path in the Ply graph.
    Each record is a string in the format: "move,side,agn,points,score,visits"
    """
    records = []

    def traverse(ply: NativePly, path: CSVRecord):
        global row_no
        if ply != ply_root:
            path.append(ply[0])
        if len(ply) > 1 and ply[1]:
            for next_ply in ply[1:]:
                traverse(next_ply, path.copy())
        else:
            records.append(",".join([f"{row_no + 1}{file_nr}"] + path))
            row_no += 1

    traverse(ply_root, [])
    return records


def main(files) -> int:
    global row_no
    file_nr = 0
    for file in files:
        row_no = 0
        with open(file, 'r') as fr:
            data = json.load(fr)
            print("File part", chr(ord('A') + file_nr))
            with open(f"{file}.csv", "w") as fw:
                fw.write("\n".join(
                    compact3ply2json(data, chr(ord('A') + file_nr))
                ))
        file_nr += 1
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
1