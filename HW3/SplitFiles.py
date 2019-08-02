def parse_single_file(lst):
    doc_file = None
    for i in lst:
        if i.startswith('*TEXT'):
            if doc_file is not None:
                doc_file.close()
            d_file = int(i.split()[1])
            doc_file = open('doc/Doc_' + str(d_file) + '.txt', 'w')
        elif i != '\n':
            doc_file.write(i)

orig_file = './time/TIME.ALL'
files = open(orig_file, 'r')

a = files.readlines()
parse_single_file(a)

