import csv


def read_fie_rows(file_path):
    rows = []
    tsv_file = open(file_path)
    tsv_reader = csv.reader(tsv_file, delimiter="\t")

    for row in tsv_reader:
        rows.append(row)
    return rows


file_rows = read_fie_rows('../dataset/dontpatronizeme_categories.tsv')

categories = set()
for curr_row in file_rows[5:]:
    categories.add(curr_row[3])

print(len(file_rows))
print(list(categories))

file_rows = read_fie_rows('../dataset/dontpatronizeme_categories.tsv')
for curr_row in file_rows[5:10000]:
    if curr_row[-2] == 'The_poorer_the_merrier':
        print(curr_row)
