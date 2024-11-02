#!/usr/bin/env python3

import sys
import json
from deepdiff import DeepDiff


def main(file1, file2):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    diffs = DeepDiff(data1, data2)
    if diffs:
        import pudb; pu.db
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
