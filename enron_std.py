import glob
import os
from configuration_vars import *
from sklearn.model_selection import train_test_split
import numpy as np
import std
import random
from sklearn import preprocessing
from pathlib import Path


def get_author_name_from_folder(dir):
    return dir[:-3]


def get_author_name_from_unknown_file(name):
    return name[9:].split(' ')[0]


def read_file(file, dir):
    return Path(DATASET_PATH + dir + os.sep + file).read_text()


def get_dataset():
    known_files = []
    known_files_grouped_by_author = []
    unknown_files = []
    unknown_authors = []
    author = []

    authors_dirs = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]

    for dir in authors_dirs:
        files = os.listdir(DATASET_PATH + dir + os.sep)
        tmp_known_files = np.array(list(filter(lambda x: x.startswith('known'), files)))

        for kfile in tmp_known_files:
            known_files.append(read_file(kfile, dir))

        # grouped label
        author.extend(len(tmp_known_files) * [dir])
        concatenarted_file = [" ".join([read_file(t, dir) for t in tmp_known_files])]
        known_files_grouped_by_author.append(concatenarted_file[0])

        # test set
        unk_tmp = list(filter(lambda x: x.startswith('unknown'), files))
        unknown_files.append(np.array(read_file(unk_tmp[0], dir)))
        unknown_authors.append(get_author_name_from_unknown_file(unk_tmp[0]))

    author_grouped = [get_author_name_from_folder(n) for n in authors_dirs]
    author = [get_author_name_from_folder(n) for n in author]
    return np.array(known_files), np.array(known_files_grouped_by_author), np.array(author), np.array(author_grouped), np.array(unknown_files), np.array(unknown_authors)


def random_shuffle(known_files, known_files_grouped, authors, authors_grouped, seed=4):
    z = zip(known_files, known_files_grouped, authors, authors_grouped)
    random.Random(seed).shuffle(z)
    return zip(*z)


# returns:
# single_author, grouped_authors
# (X_train, X_test, y_train, y_test), (X_train_g, X_test_g, y_train_g, y_test_g)
def split_train_test(known_files, known_files_grouped, authors, authors_grouped, test_size=0.33):
    known_files, known_files_grouped, authors, authors_grouped = random_shuffle(known_files, known_files_grouped, authors, authors_grouped)

    X_train, X_test, y_train, y_test = train_test_split(known_files, authors, test_size=test_size, shuffle=False)
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(known_files_grouped, authors_grouped, test_size=test_size, shuffle=False)

    return (X_train, X_test, y_train, y_test), (X_train_g, X_test_g, y_train_g, y_test_g)


known_files, known_files_grouped, authors, authors_grouped, unknown_files, unknown_authors = get_dataset()
print(known_files.shape, known_files_grouped.shape, authors.shape, authors_grouped.shape, unknown_files.shape, unknown_authors.shape)

le = preprocessing.LabelEncoder()
le.fit(authors)

print(unknown_authors.shape)

authors = le.transform(authors)
authors_grouped = le.transform(authors_grouped)
unknown_authors = le.transform(unknown_authors)

print(unknown_files)

# single_author := (X_train, X_test, y_train, y_test),
# grouped_authors := (X_train_g, X_test_g, y_train_g, y_test_g)
# single_author, grouped_authors = split_train_test(known_files, known_files_grouped, authors, authors_grouped, test_size=0.33)

# train_data_c, test_data_c = char_gram()
# print(b.shape)
# print(c.shape)
# print(d.shape)
