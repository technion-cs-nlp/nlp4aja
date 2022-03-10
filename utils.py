import os
import pickle
from sklearn.model_selection import train_test_split, KFold
from numpy import array
import random
import json
import argparse
from collections import Counter
from copy import deepcopy
import math


def split_train_test_val(all_sentences_paths, output_dir, random_seed=0, max_words=math.inf):
    sentences, spare_sentences = get_sentences_from_pickle_list(all_sentences_paths, max_words)
    train, test_val = train_test_split(sentences, test_size=0.2, random_state=random_seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=random_seed)
    with open(os.path.join(output_dir, "train.pkl"), 'wb') as f_train:
        pickle.Pickler(f_train, protocol=-1).dump(train)
        print("wrote train sentences to train.pkl")
    with open(os.path.join(output_dir, "test.pkl"), 'wb') as f_test:
        pickle.Pickler(f_test, protocol=-1).dump(test)
        print("wrote test sentences to test.pkl")
    with open(os.path.join(output_dir, "val.pkl"), 'wb') as f_val:
        pickle.Pickler(f_val, protocol=-1).dump(val)
        print("wrote val sentences to val.pkl")
    with open(os.path.join(output_dir, "spare.pkl"), 'wb') as f_spare:
        pickle.Pickler(f_spare, protocol=-1).dump(spare_sentences)
    return train, test, val


def split_train_val(all_sentences_paths, output_dir, random_seed=0, max_words=math.inf):
    sentences, spare_sentences = get_sentences_from_pickle_list(all_sentences_paths, max_words)
    train, val = train_test_split(sentences, test_size=0.1, random_state=random_seed)
    train_path = os.path.join(output_dir, "train.pkl")
    with open(train_path, 'wb') as f_train:
        pickle.Pickler(f_train, protocol=-1).dump(train)
        print("wrote train sentences to train.pkl")
    val_path = os.path.join(output_dir, "val.pkl")
    with open(val_path, 'wb') as f_val:
        pickle.Pickler(f_val, protocol=-1).dump(val)
        print("wrote val sentences to val.pkl")
    with open(os.path.join(output_dir, "spare.pkl"), 'wb') as f_spare:
        pickle.Pickler(f_spare, protocol=-1).dump(spare_sentences)
    return train, val, train_path, val_path


def k_fold(all_sentences_paths, k=10, random_seed=None, shuffle=True, max_words=math.inf):
    sentences, spare_sentences = get_sentences_from_pickle_list(all_sentences_paths, max_words)
    
    if random_seed is not None:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=k)
    random.seed(random_seed)
    for train, val in kf.split(sentences):
        train_cpy = deepcopy(train)
        val_cpy = deepcopy(val)
        if shuffle:
            random.shuffle(train_cpy)
            random.shuffle(val_cpy)
        yield array(sentences)[train_cpy], array(sentences)[val_cpy]
        
        
def get_sentences_from_pickle_list(pickle_list, max_words):
    all_sentences = []
    for i, pickle_path in enumerate(pickle_list):
        with open(pickle_path, 'rb') as f:
            all_sentences.extend(pickle.load(f))
    shuffled_sentences = random.sample(all_sentences, len(all_sentences))
    running_word_sum = 0
    returned_sentences = []
    i = 0
    while running_word_sum <= max_words and i < len(shuffled_sentences):
        returned_sentences.append(shuffled_sentences[i])
        running_word_sum += len(shuffled_sentences[i].word_analyses)
        i += 1
    if i < len(shuffled_sentences):
        spare_sentences = shuffled_sentences[i:]
    else:
        spare_sentences = []
    return returned_sentences, spare_sentences


def find_crossing_duplicates(all_sentences_path, k, random_seed, dup_list_path, shuffle=True):
    with open(dup_list_path, 'r', encoding='utf8') as f:
        problem_files = json.load(f)
    fold_res = {"train_val": [], "test_val": [], "train_test": []}
    for train_val, test in k_fold(all_sentences_path, k=k, random_seed=random_seed, shuffle=shuffle):
        train = train_val[:int(len(train_val)*0.9)]
        val = train_val[int(len(train_val)*0.9):]
        print("Train length: {}\nVal length: {}\nTest length: {}".format(len(train), len(val), len(test)))
        # for sent in val:
        #     print(sent.source)
        train_val_cross = 0
        train_test_cross = 0
        test_val_cross = 0
        for val_sentence in val:
            # check if in test, inc
            if val_sentence.source[0] in problem_files:
                for test_sentence in test:
                    if test_sentence.source[0] == val_sentence.source[0]:
                        val_source = val_sentence.source[1].split(";")
                        test_source = test_sentence.source[1].split(";")
                        if val_source[0] != test_source[0] and val_source[1:] == test_source[1:]:
                            test_val_cross += 1
                            break
                # check if in train, inc
                for train_sentence in train:
                    if train_sentence.source[0] == val_sentence.source[0]:
                        val_source = val_sentence.source[1].split(";")
                        train_source = train_sentence.source[1].split(";")
                        if val_source[0] != train_source[0] and val_source[1:] == train_source[1:]:
                            train_val_cross += 1
                            break
        for test_sentence in test:
            if test_sentence.source[0] in problem_files:
                # check if in train, inc
                for train_sentence in train:
                    if train_sentence.source[0] == test_sentence.source[0]:
                        test_source = test_sentence.source[1].split(";")
                        train_source = train_sentence.source[1].split(";")
                        if test_source[0] != train_source[0] and test_source[1:] == train_source[1:]:
                            train_test_cross += 1
                            break
        fold_res["train_val"].append(train_val_cross)
        fold_res["test_val"].append(test_val_cross)
        fold_res["train_test"].append(train_test_cross)
    return fold_res


def get_pickle_stats(pickle_path, genre_map_path=None):
    with open(pickle_path, 'rb') as f:
        sentences = pickle.load(f)
    num_sents = len(sentences)
    num_words = sum([len(sent.word_analyses) for sent in sentences])
    word_types = set()
    for sent in sentences:
        for word_analysis in sent.word_analyses:
            word_types.add(word_analysis.word)
    print(f"{pickle_path} has {num_sents} sentences with {num_words} word tokens and {len(word_types)} word types.")
    if genre_map_path:
        with open(genre_map_path, 'r', encoding='utf8') as genre_f:
            genre_map = json.load(genre_f)
        file_to_genre = {}
        for genre, files in genre_map.items():
            for file in files:
                file_to_genre[file] = genre
        genre_counter = Counter()
        for sent in sentences:
            sent_genre = file_to_genre[sent.source[0]]
            genre_counter[sent_genre] += len(sent.word_analyses)
        for genre, count in genre_counter.items():
            print(f"{genre} has {count} words, {count/num_words} of total pickle")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run utils')
    parser.add_argument('--all_sentences_path', help="path to sentence pickle")
    parser.add_argument('--dup_list_path', help="path to sentence pickle")
    parser.add_argument('-k', type=int, help="path to sentence pickle")
    parser.add_argument('--shuffle', action="store_true")
    
    parser.add_argument('--get_stats', action="store_true")
    parser.add_argument('--genre_map_path')

    args, unknown = parser.parse_known_args()
    if args.dup_list_path:
        print(find_crossing_duplicates(args.all_sentences_path, args.k, 0,
                                       args.dup_list_path, shuffle=args.shuffle))
    if args.get_stats:
        get_pickle_stats(args.all_sentences_path, genre_map_path=args.genre_map_path)
