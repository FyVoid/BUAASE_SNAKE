filenames = [
    "fc1.weight.txt", "fc1.bias.txt", "fc2.weight.txt", "fc1.bias.txt", "fc3.weight.txt", "fc1.bias.txt"
]

for filename in filenames:
    out = filename + ".out"
    of = open(out, "w")
    of.write("{\n")

    with open(filename, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == len(lines) - 1:  # Check if it's the last line
                of.write(str(float(line.strip())) + "\n}\n")
            else:
                of.write(str(float(line.strip())) + ", ")
    
