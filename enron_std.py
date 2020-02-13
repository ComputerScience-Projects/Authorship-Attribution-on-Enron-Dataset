import glob
import os
from time import time

from configuration_vars import *
from sklearn.model_selection import train_test_split
import numpy as np
import std
import random
from sklearn import preprocessing
from pathlib import Path
from classifier import *
from sklearn.metrics import precision_score, accuracy_score, classification_report
import matplotlib.pyplot as plt


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
    return np.array(known_files), np.array(known_files_grouped_by_author), np.array(author), np.array(
        author_grouped), np.array(unknown_files), np.array(unknown_authors)


def random_shuffle(known_files, known_files_grouped, authors, authors_grouped, seed=4):
    z = zip(known_files, known_files_grouped, authors, authors_grouped)
    random.Random(seed).shuffle(z)
    return zip(*z)


# returns:
# single_author, grouped_authors
# (X_train, X_test, y_train, y_test), (X_train_g, X_test_g, y_train_g, y_test_g)
def split_train_test(known_files, known_files_grouped, authors, authors_grouped, test_size=0.33):
    known_files, known_files_grouped, authors, authors_grouped = random_shuffle(known_files, known_files_grouped,
                                                                                authors, authors_grouped)

    X_train, X_test, y_train, y_test = train_test_split(known_files, authors, test_size=test_size, shuffle=False)
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(known_files_grouped, authors_grouped,
                                                                test_size=test_size, shuffle=False)

    return (X_train, X_test, y_train, y_test), (X_train_g, X_test_g, y_train_g, y_test_g)


def metrics(true, pred):
    print(precision_score(true, pred, average=None))
    print(precision_score(true, pred, average='micro'))
    print(accuracy_score(true, pred, normalize=False))
    print(classification_report(true, pred))


def evaluate_problem_net(texts, problem: str, language, path, pt):
    t0 = time()

    train_texts, train_labels, test_texts, candidate_grouped_texts = texts

    print("train and test time():", time() - t0)
    # std.print_problem_data(language, index, candidates, train_texts, test_texts)

    train_data, test_data, cosine_matrix_test_c = vectorization(train_texts, test_texts, path, problem, language,
                                                                candidate_grouped_texts)
    # train_data e' la lista dei training data, una lista di matrici

    # train_data, test_data = dimentionality_reduction(train_data, test_data)

    train_data, test_data = std.scale(train_data, test_data, print_time=True)

    classification_func = classification
    if MULTICLASSIFIER: classification_func = multi_classification
    predictions_list, proba_list = classification_func(train_data, test_data, path, problem, language, train_labels, 0, 0, 0)
    # prediction e' la matrice delle probabilità delle predizioni

    for predictions, proba in zip(predictions_list, proba_list):

        # Reject for unk classification
        if cosine_matrix_test_c is None:
            predictions = std.reject_option(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        else:
            predictions = std.reject_option_cosine(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        # stats_data = std.save_output(path, problem, unk_folder, predictions, outpath, proba)

    print(predictions)
    return predictions


def evaluate_problem(texts, problem: str, language, path, pt):
    print(problem, language)
    t0 = time()
    # candidates, unk_folder = std.get_problem_info(path, problem)

    # train_docs, train_texts, train_labels, test_texts, candidate_grouped_texts = std.get_train_andtest_set(candidates, path, problem, unk_folder, pickle_path,
    #                                                                                                       use_storage=False and S)

    train_texts, train_labels, test_texts, candidate_grouped_texts = texts

    print("train and test time():", time() - t0)
    # std.print_problem_data(language, index, candidates, train_texts, test_texts)

    train_data, test_data, cosine_matrix_test_c = vectorization(train_texts, test_texts, path, problem, language,
                                                                candidate_grouped_texts)

    # train_data, test_data = dimentionality_reduction(train_data, test_data)

    train_data, test_data = std.scale(train_data, test_data, print_time=True)

    #  print("data_shape", train_data[0].shape)

    #  for i in range(len(train_data)):
    #    kselector = SelectKBest(chi2, k=k)
    #    train_data[i] = kselector.fit_transform(train_data[i], train_labels)
    #    test_data[i] = kselector.transform(test_data[i])
    classification_func = classification
    if MULTICLASSIFIER: classification_func = multi_classification
    predictions_list, proba_list = classification_func(train_data, test_data, path, problem, language, train_labels, 0,
                                                       0, 0)

    # print(np.array(test_data).shape)
    # predictions_list, proba_list = std.compression_evaluation(test_data)
    # print(predictions_list)
    # print(np.array(proba_list).shape)

    # cosine_matrix_test_c = None

    # unk_predictions_list, unk_proba_list = unk_classification(train_data, test_data, path, problem)

    for predictions, proba in zip(predictions_list, proba_list):

        if cosine_matrix_test_c is None:
            predictions = std.reject_option(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        else:
            predictions = std.reject_option_cosine(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        # stats_data = std.save_output(path, problem, unk_folder, predictions, outpath, proba)

    print(predictions)
    return predictions


#
# Preparation

known_files, known_files_grouped, authors, authors_grouped, unknown_files, unknown_authors = get_dataset()
print(known_files.shape, known_files_grouped.shape, authors.shape, authors_grouped.shape, unknown_files.shape,
      unknown_authors.shape)

le = preprocessing.LabelEncoder()
le.fit(authors)
# print(len(set(unknown_authors).intersection(set(authors_grouped))), len(authors_grouped))

authors = le.transform(authors)
authors_grouped = le.transform(authors_grouped)
unknown_authors = le.transform(unknown_authors)

#
# Run AA Classifier

texts = (known_files, authors, unknown_files, known_files_grouped)
# pred = evaluate_problem(texts, "0", 'english', "", 0.1)

#
# Metrics

pred = [24, 43, 61, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 10, 11, 12, 13, 41, 15, 16, 77, 18, 19, 69, 21, 22, 23, 25, 43, 25, 28, 52, 30, 31, 10, 33, 34, 35, 36, 37, 38, 39, 41, 37, 44, 45, 40, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 41, 52, 20, 61, 26, 63, 64, 65, 66, 67, 68, 69, 70, 24, 35, 43, 74, 37, 76, 24, 29, 79, 40]

metrics(unknown_authors, pred)

print(unknown_authors)

# single_author := (X_train, X_test, y_train, y_test),
# grouped_authors := (X_train_g, X_test_g, y_train_g, y_test_g)
# single_author, grouped_authors = split_train_test(known_files, known_files_grouped, authors, authors_grouped, test_size=0.33)

# train_data_c, test_data_c = char_gram()
# print(b.shape)
# print(c.shape)
# print(d.shape)
