#!/usr/bin/env python3

import sys


def main(file1, file2):
    with open(file1, 'r') as f:
        data1 = f.readlines()
    with open(file2, 'r') as f:
        data2 = f.readlines()
    if data1 == data2:
        print("Equal")
        return 0
    print("Different")
    for i in range(min(len(data1), len(data2))):
        if len(data1[i]) != len(data2[i]):
            print(f"Rows {i} have different number of characters: {len(data1[i])} != {len(data2[i])}")
        if ((cols1 := len(data1[i].split(","))) !=
            (cols2 := len(data2[i].split(",")))):
            print(f"Rows {i} have different number of columns: "
                  f"{cols1} != {cols2}")
        else:
            cells2 = data2[i].split(",")
            for col_no, cell1 in enumerate(data1[i].split(",")):
                if col_no > 0 and cell1 != cells2[col_no]:
                    print(f"Line {i + 1}, column {col_no + 1}:")
                    print(f"  {file1}: {cell1}")
                    print(f"  {file2}: {cells2[col_no]}")
        # if data1[i] != data2[i]:
        #     print(f"Line {i + 1}:")
        #     print(f"  {file1}: {data1[i].strip()}")
        #     print(f"  {file2}: {data2[i].strip()}")
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
