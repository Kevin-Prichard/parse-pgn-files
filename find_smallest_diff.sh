#!/usr/bin/env bash

# error
# do the JSON ply to CSV trick, so it's easier to visually compare the two files

TMPOUT="/home/kev/projs/chessan/parse-pgn-files/tmp"
FILENO=`date +%Y%m%d_%H%M%S`
skip=$1 || 0
base=$2 || ""
while [ 1 ]; do
    file1="${TMPOUT}/tree${FILENO}a.json"
    file2="${TMPOUT}/tree${FILENO}b.json"
    ./parse_pgn.py -f ./lichess_db_test.pgn -o $file1 -l 50 -p 5 -q 5 -k $skip
    ./parse_pgn.py -f ./lichess_db_test.pgn -o $file2 -l 50 -p 5 -q 5 -k $skip
    diff -q $file1 $file2 >/dev/null 2>&1
    if [ $? -eq 1 ]; then
        ./diff2json.py $file1 $file2
        if [ $? -eq 1 ]; then
            echo "Found smallest diff at $skip"
            ls -l $file1 $file2
            break
        fi
    fi
    skip=$((skip+1))
done
