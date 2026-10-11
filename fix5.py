import re
with open("src/foundry/reports/__init__.py", "r") as f:
    content = f.read()
content = content.replace('return "\\n\n", 'return "\\n".join(lines), ')
content = content.replace('",\n"    cand = candidates.get', '",\n"    cand = candidates.get')
lines = content.split("\n")
for i, line in enumerate(lines):
    if line.rstrip().endswith('"return "\\n"') or (line.rstrip() == 'return "\\n"'):
        lines[i] = line.replace('return "\\n"', 'return "\\n".join(lines)')
    if line == '"' and i > 0 and "return" in lines[i-1]:
        lines[i] = ""
content = "\n".join(lines)
with open("src/foundry/reports/__init__.py", "w") as f:
    f.write(content)
print("done")
