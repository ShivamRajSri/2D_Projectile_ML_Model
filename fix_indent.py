import sys

with open("main.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 58 <= i <= 61:
        if "p.add_argument(\"--demo\"" in line:
            lines.insert(i+1, '    p.add_argument("--interactive", action="store_true", help="Use manual interactive point selection instead of RANSAC")\n')
            break

for i in range(177, 274):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

with open("main.py", "w") as f:
    f.writelines(lines)
