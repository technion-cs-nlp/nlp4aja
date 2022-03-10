from utils import k_fold

import pickle
import os
import json
import argparse


def prepare_train_data(pickle_paths, out_folder=None, sources=False):
    """
    Receives list of paths to pickled Sentences, and returns the training data in expected format
    If sources flag is turned on, it also returns a list of the source file of each sentence, in
    the same order as the training sentences appear
    :param pickle_paths:
    :param out_folder:
    :param sources:
    :return:
    """
    train_sentences = []
    for pickle_path in pickle_paths:
        with open(pickle_path, 'rb') as f:
            train_sentences.extend(pickle.load(f))
    sentences, word_count = prepare_train_sentences(train_sentences)
    sent_sources = [sent.source for sent in train_sentences]
    if out_folder:
        with open(os.path.join(out_folder, "train.json"), 'w+', encoding='utf8') as f:
            json.dump(sentences, f, ensure_ascii=False)
    if sources:
        return sentences, word_count, sent_sources
    return sentences, word_count


def prepare_test_data(pickle_paths, out_dir=None, sources=True):
    sentences_and_others = prepare_train_data(pickle_paths, sources=sources)
    if out_dir:
        with open(os.path.join(out_dir, "test.json"), 'w+', encoding='utf8') as f:
            json.dump(sentences_and_others[0], f, ensure_ascii=False)
    # if sources=True, returns sentences and sources; otherwise, returns sentences and word count
    return sentences_and_others[0], sentences_and_others[-1]


def prepare_untagged_data(pickle_paths, out_dir=None):
    untagged_sentence_objects = []
    for pickle_path in pickle_paths:
        with open(pickle_path, 'rb') as f:
            untagged_sentence_objects.extend(pickle.load(f))
    sentences, _ = prepare_train_sentences(untagged_sentence_objects)
    untagged_sentences = [sentence for (sentence, tags) in sentences]
    to_remove = set()
    for i, sentence in enumerate(untagged_sentences):
        if len(sentence) == 0:
            to_remove.add(i)
    untagged_sentence_objects = [sent for i, sent in enumerate(untagged_sentence_objects) if i not in to_remove]
    untagged_sentences = [sent for i, sent in enumerate(untagged_sentences) if i not in to_remove]
    if out_dir:
        with open(os.path.join(out_dir, "untagged.json"), 'w+', encoding='utf8') as f:
            json.dump(untagged_sentences, f, ensure_ascii=False)
    return untagged_sentences, untagged_sentence_objects


def prepare_train_sentences(train_sentences):
    sentences = []
    word_count = {}
    for sentence in train_sentences:
        sent = [(analysis.word, analysis.bpe) for analysis in sentence.word_analyses]
        tags = [(analysis.pos, analysis.analysis1, analysis.analysis2,
                 analysis.analysis3, analysis.enclitic_pronoun)
                for analysis in sentence.word_analyses]
        for word, _ in sent:
            word_count[word] = word_count.get(word, 0) + 1
        sentences.append((sent, tags))
    return sentences, word_count


def prepare_kfold_data(pickle_paths, k=10, seed=None):
    for train, test in k_fold(pickle_paths, k=k, random_seed=seed):
        train_sentences, train_word_count = prepare_train_sentences(train)

        test_sentences, _ = prepare_train_sentences(test)
        yield train_sentences[:int(len(train_sentences) * 0.9)], \
              train_sentences[int(len(train_sentences) * 0.9):], test_sentences, \
              train_word_count


def split_pickle_by_heldout_genres(pickle_path, genre_to_file_path, specified_genres, print_stats=False):
    with open(genre_to_file_path, 'r', encoding='utf8') as f:
        genre_to_file_map = json.load(f)
    for specified_genre in specified_genres:
        if specified_genre not in genre_to_file_map:
            print(f"{specified_genre} is not in the provided list of genres.")
            return [], []
    with open(pickle_path, 'rb') as pkl_f:
        all_sentences = pickle.load(pkl_f)
    rest_of_sents = []
    specified_genre_sents = []
    for sentence in all_sentences:
        for specified_genre in specified_genres:
            if sentence.source[0] in genre_to_file_map[specified_genre]:
                specified_genre_sents.append(sentence)
            else:
                rest_of_sents.append(sentence)
    if print_stats:
        num_train_sents = len(rest_of_sents)
        num_held_out_sents = len(specified_genre_sents)
        num_train_words = sum([len(sent.word_analyses) for sent in rest_of_sents])
        num_held_out_words = sum([len(sent.word_analyses) for sent in specified_genre_sents])
        total_words = num_held_out_words + num_train_words
        print(f"Held out data has {num_held_out_sents} sentences and {num_held_out_words} words")
        print(f"Rest of data has {num_train_sents} sentences and {num_train_words} words")
        print(f"Train/test ratio (words): {num_train_words / total_words}/{num_held_out_words / total_words}")
    return rest_of_sents, specified_genre_sents


def split_pickle_into_genres(pickle_path, out_dir, genre_to_file_path, print_stats=False):
    with open(genre_to_file_path, 'r', encoding='utf8') as f:
        genre_to_file_map = json.load(f)
    file_to_genre = {}
    for genre, files in genre_to_file_map.items():
        for file in files:
            file_to_genre[file] = genre
    with open(pickle_path, 'rb') as pkl_f:
        all_sentences = pickle.load(pkl_f)
    genre_to_genre_sents = {genre: [] for genre in genre_to_file_map.keys()}
    for sentence in all_sentences:
        genre = file_to_genre[sentence.source[0]]
        genre_to_genre_sents[genre].append(sentence)
    for genre, sentences in genre_to_genre_sents.items():
        outpath = os.path.join(out_dir, genre + ".pkl")
        if len(sentences) > 0:
            with open(outpath, 'wb') as f:
                pickle.dump(sentences, f)
        if print_stats:
            num_words = sum([len(sent.word_analyses) for sent in sentences])
            num_sentences = len(sentences)
            print(f"Genre {genre} has {num_sentences} sentences and {num_words} words")


def main(args):
    if args.split_by_genre:
        split_pickle_into_genres(args.all_sentences_path, args.outdir, args.genre_map_path, args.get_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split pickle by genres')
    parser.add_argument('--all_sentences_path', help="path to sentence pickle")
    parser.add_argument('--outdir', help="path to directory for saving outputted pickles")
    parser.add_argument('--split_by_genre', action="store_true", help="Save each genre to separate pickle")
    parser.add_argument('--genre_map_path')
    parser.add_argument('--train_genres', nargs='+')
    parser.add_argument('--get_stats', action="store_true")

    args, unknown = parser.parse_known_args()
    main(args)
