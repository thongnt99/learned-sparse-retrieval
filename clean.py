from tqdm import tqdm

import sys

file = sys.argv[1]

def clean_file(file):

    new_lines = list()

    with open(file) as f:
        for line in tqdm(f):
            splitted = line.strip().split(" ")
            qid = splitted[0]
            did = splitted[2]
            if qid == did:
                continue
            else:
                new_lines.append(line)
    
    with open(file+".fixed","w") as f:
        for line in tqdm(new_lines):
            f.write(line)

clean_file(file)