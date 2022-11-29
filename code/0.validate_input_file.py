"""
Checks that the file's tagset matches the expected one.
"""

import sys


def validate(in_file, tagset_file):
    tags = set()
    with open(tagset_file, encoding="utf8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.add(tag)
    with open(in_file, encoding="mac-roman") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            cells = line.split("\t")
            if len(cells) != 2:
                print("Expected two cells, but found " + len(cells))
                print(line)
                return False
            if cells[1] not in tags:
                print("Not a valid tag: " + cells[1])
                print(line)
                return False
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 0.validate_input_file.py INPUT_FILE TAGSET_FILE")
        sys.exit(0)
    success = validate(sys.argv[1], sys.argv[2])
    if success:
        print("Validation successful!")
