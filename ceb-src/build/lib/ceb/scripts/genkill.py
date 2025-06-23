import re
fin = open("pypid", "r")
fout = open("killpy", "w")

pat = re.compile(r"^(lirui|kfzhao|shfang)\s+([0-9]+)\s+.+python3\s-u\s\s*.*$")
pat2 = re.compile(r"^.+grep.+$")

while True:
    line = fin.readline()
    if not line:
        break
    line = line.strip("\n")
    m = pat.match(line)
    m2 = pat2.match(line)
    if m and not m2:
        fout.write("kill " + str(m.group(2)) + "\n")
