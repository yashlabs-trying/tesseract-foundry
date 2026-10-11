with open("src/foundry/reports/__init__.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    # Check if line ends with return "\\n
    stripped = line.rstrip()
    if stripped == chr(9)*8 + 'return "\\n' or stripped == chr(9)*4 + 'return "\\n':
        new_lines.append(stripped + '.join(lines)' + '\n')
        # skip the next empty line and ".join(lines) if present
        if i+1 < len(lines) and lines[i+1].strip() in ('",', '".join(lines)', ''):
            skip_next = True
        if i+2 < len(lines) and lines[i+2].strip() == '".join(lines)':
            skip_next = True
    else:
        new_lines.append(line)

with open("src/foundry/reports/__init__.py", "w") as f:
    f.writelines(new_lines)
print("done")
