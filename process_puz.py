import os
import subprocess
import get_puzpy
import sklearn
import pandas as pd

has_puzpy = os.path.isdir('puzpy')

if not has_puzpy:
    get_puzpy.get_puzpy()

import puzpy.puz as puz


class Puz:
    def __init__(self, file):
        self.file = puz.read(file)
        p = self.file
        self.numbering = numbering = p.clue_numbering()
        with open(file, encoding='iso8859-1') as f:
            self.puz_text = [line for line in f]
            self.puz_text = self.puz_text[0]
        self.rows = self.get_across()
        self.cols = self.get_down()
        self.answers = self.rows, self.cols
    def get_across(self):
        rows = []
        for clue in self.numbering.across:
            answer = ''.join(self.file.solution[clue['cell'] + i] for i in range(clue['len']))
            rows.append((clue['num'], 'Across', clue['clue'], '-', answer))
        return rows

    def get_down(self):
        cols = []
        for clue in self.numbering.down:
            answer = ''.join(self.file.solution[clue['cell'] + i * self.numbering.width] for i in range(clue['len']))
            cols.append((clue['num'], 'Down', clue['clue'], '-', answer))
        return cols

    def print_all(self):
        for ans in self.answers:
            for a in ans:
                print(a)

    

# testing
# file = 'nyt/daily/2008/02/Feb0108.puz'

# my_puz = Puz(file)
# my_puz.print_all()

def iterate_through_archive():
    archive = []
    root = 'nyt/daily'
    for year in os.listdir(root):
        for month in os.listdir(f'{root}/{year}'):
            for puzzle in os.listdir(f'{root}/{year}/{month}'):
                archive.append(f'{root}/{year}/{month}/{puzzle}')
    return archive

#TODO: write function that replaces hint references with that hint's question
#for hint in answer:
#   if {regex} in hint:
#       ref_clue = answer.find()
#       hint = hint - regex + ref_sclue


def find_max_len(archive):
    max_len = 0
    for question in archive:
        if len(question[2]) > max_len:
            max_len = len(question[2])

def string_to_csv(archive, filename):
    labels = "ID,Num,Dir,Hint,Ans"
    with open(filename, 'w') as f:
        f.write(labels + '\n')
        for clue in archive:
            #print(clue)
            f.write(f'{archive.index(clue)},{clue[0]},{clue[1]},{clue[2]},{clue[4]}\n')

def archive_to_string():
    arch_strings = []
    temp_archive = iterate_through_archive()
    for entry in temp_archive:
        try:
            current_puz = Puz(entry)
            for ans in current_puz.answers:
                for a in ans:
                    arch_strings.append(a)
        except puz.PuzzleFormatError:
            pass
    return arch_strings

def csv_to_tsv(csv_file):
    buffer = []
    header = True
    with open(csv_file) as f, open(f'{str(csv_file)[:-4]}.tsv', 'w') as g:
        for line in f:
            if header:
                header = False
                g.write('ID\tNum\tDir\tClue\tAns\tLen\n')
            else:
                # Split the line into a list using commas as separators
                values = line.split(',')
                # if len(values) > 6:
                #     print(len(values))
                # Replace the first, second, third, and last commas with tabs
                values[0] = values[0].replace(',', '\t', 1)
                values[1] = values[1].replace(',', '\t', 1)
                values[2] = values[2].replace(',', '\t', 1)
                if len(values) == 5:
                    values[3] = values[3] + f' (Answer is {len(values[-1]) - 1} squares)'
                else:
                    values[-2] = values[-2] + f' (Answer is {len(values[-1]) - 1} squares)'
                    values[3] = ''.join(values[3:-2])
                    while len(values) > 5:
                        values.pop(4)
                values[-1] = values[-1].replace(',', '\t', 1)
                values[-1] = values[-1].replace('\n', '')
                values.append(f'{len(values[-1])}\n')

                # Join the modified values with tabs and write to the output file
                modified_line = '\t'.join(values)
                g.write(modified_line)



#testing
# test_str = archive_to_string()
# string_to_csv(test_str, 'tester.csv')
#csv_to_tsv('tester.csv')
df = pd.read_csv('tester.tsv', delimiter='\t')
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print(df.sample(10))