import sys


def check_file(filename):
    same, diff = 0, 0
    diff_cases = []
    with open(filename) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip()
            if not line:
                continue
            cells = line.split("\t")
            tok = cells[4]
            if tok and tok != "EOS":
                merged = "".join(cells[5].split("+"))
                if tok == merged:
                    same += 1
                else:
                    diff += 1
                    diff_cases.append((tok, merged, cells[5]))
    print(filename)
    print("Merging worked:", same)
    print("Merging didn't work:", diff)
    for case in diff_cases:
        print(f"tok {case[0]} merged {case[1]} segments {case[2]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one filename.")
        sys.exit(1)

    for i in range(1, len(sys.argv)):
        check_file(sys.argv[i])
