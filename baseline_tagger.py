import argparse
import json
import csv
import pickle
import os
from collections import Counter
from sklearn.metrics import classification_report
import numpy as np
import tempfile

from utils import k_fold, split_train_val


UNKNOWN = "UNKNOWN"
WORD_MOST_COMMON_POS = "most common pos"


def train(pickle_path, dict_path, morph=False):
    if morph:
        train_morph(pickle_path, dict_path)
        return
    with open(pickle_path, 'rb') as f:
        train_sentences = pickle.load(f)
    pos_count = Counter()
    word_to_pos_count = {}
    for sentence in train_sentences:
        for word_analysis in sentence.word_analyses:
            word_to_pos_count[word_analysis.word] = word_to_pos_count.get(word_analysis.word, Counter())
            word_to_pos_count[word_analysis.word][word_analysis.pos] += 1
            pos_count[word_analysis.pos] += 1
    word_to_most_freq = {word: count.most_common(1)[0]
                         for word, count in word_to_pos_count.items()}
    most_frequent_pos = pos_count.most_common(1)[0]
    word_to_most_freq[UNKNOWN] = most_frequent_pos
    with open(dict_path, 'w+', encoding='utf8') as f_train:
        json.dump(word_to_most_freq, f_train, ensure_ascii=False)


def train_morph(pickle_path, dict_path):
    with open(pickle_path, 'rb') as f:
        train_sentences = pickle.load(f)
    word_to_pos_to_morph = {}
    pos_to_morph_count = {}
    for sentence in train_sentences:
        for word_analysis in sentence.word_analyses:
            pos_to_morph_count[word_analysis.pos] = pos_to_morph_count.get(
                word_analysis.pos,
                {"analysis1": Counter(),
                 "analysis2": Counter(),
                 "analysis3": Counter(),
                 "enclitic": Counter()})
            pos_to_morph_count[word_analysis.pos]["analysis1"][word_analysis.analysis1] += 1
            pos_to_morph_count[word_analysis.pos]["analysis2"][word_analysis.analysis2] += 1
            pos_to_morph_count[word_analysis.pos]["analysis3"][word_analysis.analysis3] += 1
            pos_to_morph_count[word_analysis.pos]["enclitic"][word_analysis.enclitic_pronoun] += 1
            if word_analysis.word not in word_to_pos_to_morph:
                word_to_pos_to_morph[word_analysis.word] = {}
                word_to_pos_to_morph[word_analysis.word][word_analysis.pos] = {
                    "analysis1": Counter([word_analysis.analysis1]),
                    "analysis2": Counter([word_analysis.analysis2]),
                    "analysis3": Counter([word_analysis.analysis3]),
                    "enclitic": Counter([word_analysis.enclitic_pronoun])}

            else:
                if word_analysis.pos not in word_to_pos_to_morph[word_analysis.word]:
                    word_to_pos_to_morph[word_analysis.word][word_analysis.pos] = {
                        "analysis1": Counter([word_analysis.analysis1]),
                        "analysis2": Counter([word_analysis.analysis2]),
                        "analysis3": Counter([word_analysis.analysis3]),
                        "enclitic": Counter([word_analysis.enclitic_pronoun])}
                else:
                    word_to_pos_to_morph[word_analysis.word][
                        word_analysis.pos]["analysis1"][word_analysis.analysis1] += 1
                    word_to_pos_to_morph[word_analysis.word][
                        word_analysis.pos]["analysis2"][word_analysis.analysis2] += 1
                    word_to_pos_to_morph[word_analysis.word][
                        word_analysis.pos]["analysis3"][word_analysis.analysis3] += 1
                    word_to_pos_to_morph[word_analysis.word][
                        word_analysis.pos]["enclitic"][word_analysis.enclitic_pronoun] += 1

    word_to_most_common_pos = {}
    word_to_pos_to_most_common = {word: {pos: {}} for word, poses in word_to_pos_to_morph.items() for pos in poses.keys()}
    for word, poses in word_to_pos_to_morph.items():
        total_pos_per_word = {pos: sum([sum(counter.values())
                                        for counter in word_to_pos_to_morph[word][pos].values()])
                              for pos in poses.keys()}
        most_common_pos = Counter(total_pos_per_word).most_common(1)[0]
        word_to_most_common_pos[word] = most_common_pos
        for pos, analyses in poses.items():
            most_common = {analysis: count.most_common(1)[0] for analysis, count in analyses.items()}
            word_to_pos_to_most_common[word][pos] = most_common

    for word in word_to_pos_to_most_common:
        word_to_pos_to_most_common[word][WORD_MOST_COMMON_POS] = \
            Counter(word_to_most_common_pos[word]).most_common(1)[0][0]
    # unknowns:
    pos_count = {}
    for pos in pos_to_morph_count:
        pos_sum = 0
        for analysis, count in pos_to_morph_count[pos].items():
            pos_sum += sum(count.values())
        pos_count[pos] = pos_sum
    # pos_count = {pos: sum([sum(count.values())
    #                        for analysis, count in pos_to_morph_count[pos].items()])
    #              for pos in pos_to_morph_count}
    most_frequent_pos = Counter(pos_count).most_common(1)[0]
    most_common_general_morph = pos_to_morph_count[most_frequent_pos[0]]
    word_to_pos_to_most_common[UNKNOWN] = \
        {most_frequent_pos[0]: {analysis: most_common_general_morph[analysis].most_common(1)[0]
                                for analysis in most_common_general_morph}}
    word_to_pos_to_most_common[UNKNOWN][WORD_MOST_COMMON_POS] = most_frequent_pos[0]
    with open(dict_path, 'w+', encoding='utf8') as f_train:
        json.dump(word_to_pos_to_most_common, f_train, ensure_ascii=False)


def test(pickle_path, dict_path, result_path=None, morph=False, write_predictions=False):
    with open(pickle_path, 'rb') as f:
        test_sentences = pickle.load(f)
    with open(dict_path, 'r+', encoding='utf8') as f_dict:
        freq_dict = json.load(f_dict)
    sentences = []
    true_tags = []
    for sentence in test_sentences:
        words = [analysis.word for analysis in sentence.word_analyses]
        if morph:
            sent_tags = [(analysis.pos, analysis.analysis1, analysis.analysis2,
                          analysis.analysis3, analysis.enclitic_pronoun)
                         for analysis in sentence.word_analyses]
        else:
            sent_tags = [analysis.pos for analysis in sentence.word_analyses]
        sentences.append(words)
        true_tags.append(sent_tags)
    predicted_tags = tag_sentences(sentences, freq_dict, morph)
    if write_predictions:
        if not result_path:
            print("Must provide result path to write predictions to file")
        else:
            data = [["sentence_id", "word", "pos_true", "pos_predicted", "an1_true",
                     "an1_predicted", "an2_true", "an2_predicted", "an3_true",
                     "an3_predicted", "enc_true", "enc_predicted"]]
            for i, sentence_tags in enumerate(predicted_tags):
                assert len(sentences[i]) == len(sentence_tags)
                for word, word_true_tags, tags in zip(sentences[i], true_tags[i], sentence_tags):
                    if morph:
                        data.append([i, word, word_true_tags[0], tags[0], word_true_tags[1], tags[1],
                                     word_true_tags[2], tags[2], word_true_tags[3], tags[3],
                                     word_true_tags[4], tags[4]])
                    else:
                        data.append([i, word, word_true_tags, tags])
            predict_file = result_path.split(".")
            predict_file = predict_file[0]+"-predictions.tsv"
            with open(predict_file, 'w+', encoding='utf8', newline="") as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_writer.writerows(data)
    result = evaluate_pos(predicted_tags, true_tags, morph)
    if morph:
        mic_prec = []
        mac_prec = []
        weight_prec = []
        dict_results = [result[0], result[2], result[4], result[6], result[8]]
        string_results = [result[1], result[3], result[5], result[7], result[7]]
        for string_res, res, analysis in zip(string_results, dict_results,
                                             ["pos", "analysis1", "analysis2",
                                              "analysis3", "enclitic"]):
            if result_path:
                new_result_path = result_path.split(".")
                new_result_path = new_result_path[0]+"-"+analysis+"."+new_result_path[1]
                with open(new_result_path, 'w+', encoding='utf8') as f:
                    f.write(string_res)
            else:
                print("{} accuracy: {}".format(analysis, res["accuracy"]))
                mic_prec.append(res["accuracy"])
                mac_prec.append(res["macro avg"]["precision"])
                weight_prec.append(res["weighted avg"]["precision"])
        if not result_path:
            print("Mean accuracy (over all analyses): {}".format(np.mean(mic_prec)))
            print("Mean macro average (over all analyses): {}".format(np.mean(mac_prec)))
            print("Mean weighted average (over all analyses): {}".format(np.mean(weight_prec)))
        return
    if result_path:
        with open(result_path, 'w+', encoding='utf8') as f:
            f.write(result)
    else:
        print(result)


def tag(pickle_path, dict_path, result_path, morph=False):
    with open(pickle_path, 'rb') as f:
        sentences = pickle.load(f)
    with open(dict_path, 'r+', encoding='utf8') as f_dict:
        freq_dict = json.load(f_dict)
    # TODO - FUNCTION NOT COMPLETED


def tag_sentences(sentences, freq_dict, morph=False):
    if morph:
        predicted = tag_sents_morph(sentences, freq_dict)
        return predicted
    predicted = []
    for sentence in sentences:
        pred = []
        for word in sentence:
            pred.append(freq_dict.get(word, freq_dict[UNKNOWN])[0])
        predicted.append(pred)
    return predicted


def tag_sents_morph(sentences, freq_dict):
    predicted = []
    for sentence in sentences:
        pred = []
        for word in sentence:
            if word in freq_dict:
                pos = freq_dict[word][WORD_MOST_COMMON_POS]
                an1 = freq_dict[word][pos]["analysis1"][0]
                an2 = freq_dict[word][pos]["analysis2"][0]
                an3 = freq_dict[word][pos]["analysis3"][0]
                enc = freq_dict[word][pos]["enclitic"][0]
            else:
                pos = freq_dict[UNKNOWN][WORD_MOST_COMMON_POS]
                an1 = freq_dict[UNKNOWN][pos]["analysis1"][0]
                an2 = freq_dict[UNKNOWN][pos]["analysis2"][0]
                an3 = freq_dict[UNKNOWN][pos]["analysis3"][0]
                enc = freq_dict[UNKNOWN][pos]["enclitic"][0]

            pred.append((pos, an1, an2, an3, enc))
        predicted.append(pred)
    return predicted


def evaluate_pos(predicted, true, morph=False):
    if morph:
        true_pos = [tags[0] for sent in true for tags in sent]
        true_1 = [tags[1] for sent in true for tags in sent]
        true_2 = [tags[2] for sent in true for tags in sent]
        true_3 = [tags[3] for sent in true for tags in sent]
        true_enc = [tags[4] for sent in true for tags in sent]

        pred_pos = [pos for sent in predicted for pos, _, _, _, _ in sent]
        pred_1 = [an1 for sent in predicted for _, an1, _, _, _ in sent]
        pred_2 = [an2 for sent in predicted for _, _, an2, _, _ in sent]
        pred_3 = [an3 for sent in predicted for _, _, _, an3, _ in sent]
        pred_enc = [enc for sent in predicted for _, _, _, _, enc in sent]

        report_pos_dict = classification_report(true_pos, pred_pos, output_dict=True)
        report_pos = classification_report(true_pos, pred_pos)
        report_1_dict = classification_report(true_1, pred_1, output_dict=True)
        report_1 = classification_report(true_1, pred_1)
        report_2_dict = classification_report(true_2, pred_2, output_dict=True)
        report_2 = classification_report(true_2, pred_2)
        report_3_dict = classification_report(true_3, pred_3, output_dict=True)
        report_3 = classification_report(true_3, pred_3)
        report_enc_dict = classification_report(true_enc, pred_enc, output_dict=True)
        report_enc = classification_report(true_enc, pred_enc)

        return report_pos_dict, report_pos, report_1_dict, report_1,\
               report_2_dict, report_2, report_3_dict, report_3, \
               report_enc_dict, report_enc

    true_flat = [pos for sent in true for pos in sent]
    predicted_flat = [tags for sent in predicted for tags in sent]
    report = classification_report(true_flat, predicted_flat)
    report_dict = classification_report(true_flat, predicted_flat, output_dict=True)
    return report


def main(args):
    if args.split_all_sentences:
        split_train_val(args.pickle_path, args.split_all_sentences)
        train_pickle_path = os.path.join(args.split_all_sentences, "train.pkl")
        test_pickle_path = os.path.join(args.split_all_sentences, "val.pkl")
    else:
        train_pickle_path = args.pickle_path
        test_pickle_path = args.pickle_path
    if args.train:
        train(train_pickle_path, args.dict_path, args.morph)
    elif args.test:
        test(test_pickle_path, args.dict_path, args.result_path, args.morph, args.write_predictions)
    elif args.tag:
        tag(args.pickle_path, args.dict_path, args.result_path, args.morph)
    else:
        print("Must select either train (-r), test (-e) or tag (-a)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train, test, or calculate POS using marmot')

    parser.add_argument('pickle_path', help="Path to data to train on/test on/tag (in pickle format)")
    parser.add_argument('dict_path', help="Path to dictionary containing most frequent POS per word")
    parser.add_argument('-res', '--result_path', type=str, help="Path to save result of tagging")

    parser.add_argument(
        "--morph",
        action='store_true',
        help="train/test/tag morphological analyses"
    )
    parser.add_argument('--split_all_sentences', default="")
    parser.add_argument('--write_predictions', action='store_true', help='Write predictions to file when testing')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-r', '--train', action='store_true')
    group.add_argument('-e', '--test', action='store_true')
    group.add_argument('-a', '--tag', action='store_true')

    args, unknown = parser.parse_known_args()

    main(args)
