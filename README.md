
# parse-pgn-files

This repository is a fork, with the parser hoisted into separate files for easier utilization elsewhere.

## Original notebook
A notebook demoing how to parse Portable Game Notation (PGN) using parsita. A walkthrough of the code can be found [here](https://a-matteson.medium.com/parsing-pgn-chess-games-with-python-68a2c199665c).

## Parsita
This Python module is a parser combinator library. It allows you to easily define a grammar using simple Python expressions and assignment statements, and then parse text according to that grammar.

The library is available on PyPI, and can be installed using pip.

## PGN Specification
http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm

## Further developments
- pgn_parser.py: the Parsita grammar-parser, defined by [Andrew Matteson's](https://medium.com/@a-matteson) in his original notebook, was hoisted into this file, making it available for use elsewhere
- this PGN grammar was further developed with the additions-
  - handling and extracting move commentary ('?!', '!!', etc.)
  - converting move sequences into a list of dicts per move, with keys for move number, white move, black move, and commentary
  - handling LiChess's split move format (e.g. '1. e4  1... e5 2. Nf3 2... Nc6')
  - ensuring a PGN's game moves correctly export as JSON
- move_parser.py: a Parsita grammar-parser for breaking down algebraic notation into detailed dicts (used for board manipulation)
- parse_pgn.py: a script for parsing a PGN files and streams
  - it provides options for parsing a single PGN file, a file of multiple PGNs, or a pipe/stream of PGNs
  - in current form it generates basic statistics from counted games, such as moves
  - it provides options to run across multiple cores using the multiprocessing module
  - it can output results to a JSON file
  - it can be extended to perform more useful analyses than simple counts

### Example usage
```bash
# parse a single PGN file...
$ ./parse_pgn.py -f ../sample_pgns.pgn -o ./results-`date +%Y%m%d-%H%M%S`.json

# parse a LiChess database...
$ zstd -dc ../lichess_db_standard_rated_2019-03.pgn.zst | ./parse_pgn.py -f /dev/stdin -o ./results-`date +%Y%m%d-%H%M%S`.txt
```

## PGN file sources
1. pgnmentor.com - https://www.pgnmentor.com/files.html
2. LiChess - https://database.lichess.org/
3. Lumbra's Giga Base - https://lumbrasgigabase.com/en/
