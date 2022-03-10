import re
import sys
import os
import math
import Levenshtein
import pandas as pd

REGEX_PUNCS = re.compile(r'[\.:\?]')
REGEX_DIGITS = re.compile(r'[0-9]')

NUM_COLS = 12


class Sentence(object):

    def __init__(self, source_file, source, line, raw_text, word_analyses, split_multi=True):

        self.source = [source_file, source]  # source is "{sheet_name};{sub_source};{line_num}"
        self.line = str(line)
        self.raw_text = raw_text
        self.cleaned_text = self.clean_text(raw_text)
        self.word_analyses = word_analyses
        if split_multi:
            self.split_multi_word_phrases()

    def split_multi_word_phrases(self, to_print=False):
        """
        :param to_print: If you want to print all the cases of multi-word phrases
        :return:
        """
        for i, analysis in enumerate(self.word_analyses):
            if "_" in analysis.word:
                new_words = analysis.word.split("_")
                for j, word in enumerate(new_words):
                    if j == 0:
                        stem = analysis.stem + "_B"
                        pos = analysis.pos + "_B"
                        analysis1 = analysis.analysis1 + "_B"
                        analysis2 = analysis.analysis2 + "_B"
                        analysis3 = analysis.analysis3 + "_B"
                        enclitic_pronoun = analysis.enclitic_pronoun + "_B"
                        # TODO if for some reason this is called after BPEs
                        # TODO are already calculated, you need to separate them too
                    elif j == len(new_words) - 1:
                        stem = analysis.stem + "_E"
                        pos = analysis.pos + "_E"
                        analysis1 = analysis.analysis1 + "_E"
                        analysis2 = analysis.analysis2 + "_E"
                        analysis3 = analysis.analysis3 + "_E"
                        enclitic_pronoun = analysis.enclitic_pronoun + "_E"
                        # TODO if for some reason this is called after BPEs
                        # TODO are already calculated, you need to separate them too
                    else:
                        stem = analysis.stem + "_I"
                        pos = analysis.pos + "_I"
                        analysis1 = analysis.analysis1 + "_I"
                        analysis2 = analysis.analysis2 + "_I"
                        analysis3 = analysis.analysis3 + "_I"
                        enclitic_pronoun = analysis.enclitic_pronoun + "_I"
                        # TODO if for some reason this is called after BPEs
                        # TODO are already calculated, you need to separate them too
                    new_analysis = WordAnalysis(word, stem, pos, analysis1, analysis2,
                                                analysis3, enclitic_pronoun,
                                                analysis.comments,
                                                analysis.orthography_pronunciation)
                    self.word_analyses.insert(i+j+1, new_analysis)
                self.word_analyses.pop(i)
                if to_print:
                    print("multi-word: {}, pos {}".format(analysis.word, analysis.pos))
                    print("new words:")
                    for new_analysis in self.word_analyses[i:i+len(new_words)]:
                        print(new_analysis.word, new_analysis.pos)

    def combine_clean_and_analyzed(self):
        """
        Takes list of word analyses and list of cleaned text, and adds "empty"
        word analyses of the extra words in the appropriate location in the list
        It does not modify the member of word analyses, but rather returns a new
        list of word analyses that matches the clean words, for comparison
        ***NOTE***
        This function does not work perfectly, and still has bugs that need to be
        worked out; at this point, it's used investigatively and not for actual
        filtering, and so the bugs have not all been fixed.
        :return:
        """
        clean_words = self.cleaned_text.split()
        new_analyzed = [None]*len(clean_words)
        for analysis in self.word_analyses:
            min_dist = math.inf
            min_i = -1
            a_word = analysis.word
            for i, clean in enumerate(clean_words):
                dist = Levenshtein.distance(a_word, clean)
                if dist < min_dist:
                    min_i = i
                    min_dist = dist
                if dist == 0:
                    if new_analyzed[min_i]:
                        continue
                    new_analyzed[min_i] = analysis
                    break
            assert min_i >= 0
            new_analyzed[min_i] = analysis
        return new_analyzed

    def clean_text(self, text):
        cleaned_text = REGEX_PUNCS.sub('', text)
        cleaned_text = REGEX_DIGITS.sub('', cleaned_text)
        cleaned_text = ' '.join(cleaned_text.strip().split())
        return cleaned_text

    def is_valid_text(self):
        """ Validate that the raw text conforms to the individual words """

        clean_words = self.cleaned_text.split()
        analyzed_words = [a.word for a in self.word_analyses]
        if len(clean_words) != len(analyzed_words):
            sys.stderr.write("Current source: {}\n".format(self.source))
            sys.stderr.write("Invalid sentence: {}\n".format(self.__str__()))
            sys.stderr.write("Num clean words: {} != num analyzed words: {}\n".
                             format(len(clean_words), len(analyzed_words)))
            sys.stderr.write("Clean words {}, analyzed words: {}\n\n".
                             format(" ".join(clean_words), " ".join(analyzed_words)))
            return False
        for clean_word, analyzed_word in zip(clean_words, analyzed_words):
            if clean_word != analyzed_word:
                return False
        return True

    def __str__(self):
        return ' '.join([self.raw_text])


# COLUMN NAMES
POS = "pos"
ANALYSIS1 = "analysis1"
ANALYSIS2 = "analysis2"
ANALYSIS3 = "analysis3"
ENCLITIC = "enclitic_pronoun"
BPE = "bpe"
COMMENTS = "comments"
ORTH_PRON = "orthography_pronunciation"


class WordAnalysis(object):

    def __init__(self, word, stem, pos, analysis1, analysis2, analysis3,
                 enclitic_pronoun, comments, orthography_pronunciation):

        # TODO: break down analyses into features
        # TODO: maybe index pos (and analyses/features)
        # TODO: eliminate cases with empty pos
        if str(word) == "nan":
            # TODO decide how to process in this case
            self.word = "_".join(str(word).split(" "))
        else:
            self.word = "_".join(str(word).strip().split(" "))
        if str(stem) == "nan":  # TODO change to ROOT (stem == binyan)
            self.stem = "_"
        else:
            self.stem = str(stem).strip()
        if str(pos) == "nan":
            self.pos = "_"
        elif len(str(pos).strip()) == 0:
            self.pos = "_"
        else:
            self.pos = str(pos).strip()
        if str(analysis1) == "nan":
            self.analysis1 = "NA"
        else:
            self.analysis1 = str(analysis1).strip()

        if str(analysis2) == "nan":
            self.analysis2 = "NA"
        else:
            self.analysis2 = str(analysis2).strip()
        if str(analysis3) == "nan":
            self.analysis3 = "NA"
        else:
            self.analysis3 = str(analysis3).strip()
        self.clean_analyses()
        if str(enclitic_pronoun) == "nan":
            self.enclitic_pronoun = "NA"
        else:
            self.enclitic_pronoun = str(enclitic_pronoun).strip()
        if str(comments) == "nan":
            self.comments = "NA"
        else:
            self.comments = str(comments).strip()
        if str(orthography_pronunciation) == "nan":
            self.orthography_pronunciation = "NA"
        else:
            self.orthography_pronunciation = str(orthography_pronunciation).strip()
        # bpe always initialized to None, needs to be set manually
        self.bpe = None
        self.clean_enclitic()


    def set_val(self, column, val):
        if column == POS:
            self.pos = val
        elif column == ANALYSIS1:
            self.analysis1 = val
        elif column == ANALYSIS2:
            self.analysis2 = val
        elif column == ANALYSIS3:
            self.analysis3 = val
        elif column == ENCLITIC:
            self.enclitic_pronoun = val
        elif column == BPE:
            self.bpe = val
        else:
            sys.stderr.write('Warning: unknown column ' + column + ' in set_val()\n')

    def get_val(self, column):
        if column == POS:
            return self.pos
        elif column == ANALYSIS1:
            return self.analysis1
        elif column == ANALYSIS2:
            return self.analysis2
        elif column == ANALYSIS3:
            return self.analysis3
        elif column == ENCLITIC:
            return self.enclitic_pronoun
        elif column == BPE:
            return self.bpe
        else:
            sys.stderr.write('Warning: unknown column ' + column + ' in get_val()\n')
            return ''

    def clean_enclitic(self):
        self.enclitic_pronoun = self.enclitic_pronoun.replace("=", "-")
        self.enclitic_pronoun = self.enclitic_pronoun.replace("--", "-")
        self.enclitic_pronoun = self.enclitic_pronoun.replace("[", "")
        self.enclitic_pronoun = self.enclitic_pronoun.replace("]", "")
        self.enclitic_pronoun = self.enclitic_pronoun.replace(".", "")
        self.enclitic_pronoun = self.enclitic_pronoun.replace(";", "")
        if self.pos == "פע" or self.pos == "מל":
            self.enclitic_pronoun = self.enclitic_pronoun.replace("ח", "מ")
            self.enclitic_pronoun = self.enclitic_pronoun.replace("ד", "מ")
        else:
            self.enclitic_pronoun = self.enclitic_pronoun.replace("ח", "ק")
            self.enclitic_pronoun = self.enclitic_pronoun.replace("ד", "ק")
        if len(self.enclitic_pronoun) == 0:
            self.enclitic_pronoun = "NA"

    def clean_analyses(self):
        for column in [ANALYSIS1, ANALYSIS2, ANALYSIS3]:
            analysis = self.get_val(column)
            if analysis[-1] in [".", ",", "?", "[", "]"]:
                analysis = analysis[:-1]
            analysis = analysis.replace("-", "")
            if "יחס" in analysis:
                analysis = analysis.replace("יחס/", "")
                analysis = analysis.replace("יחס+", "")
                analysis = analysis.replace("יחס + ", "")
            self.set_val(column, analysis)

    def check_columns_off_by_one(self, legal_values_dict):
        was_off = False
        pos_set = {pos for pos in legal_values_dict["pos"].keys()}
        an1_to_pos = {}
        for k, v in legal_values_dict["pos"].items():
            for an1 in v["analysis1"]:
                if an1 in an1_to_pos:
                    an1_to_pos[an1].append(k)
                else:
                    an1_to_pos[an1] = [k]
        if self.pos not in pos_set:
            if self.stem in pos_set:  # columns are shifted to the right
                real_pos = self.stem
                real_an1 = self.pos  # TODO should you make sure that an1 is in pos set?
                real_an2 = self.analysis1
                real_an3 = self.analysis2
                real_enc = self.analysis3
                was_off = True

            elif self.analysis1 in pos_set:  # columns are shifted to the left
                real_pos = self.analysis1
                real_an1 = self.analysis2
                real_an2 = self.analysis3
                was_off = True
                if self.enclitic_pronoun not in legal_values_dict["enclitic"]:
                    real_an3 = self.enclitic_pronoun
                    real_enc = self.comments
                else:
                    real_an3 = self.analysis3  # TODO you might have a bug here
                    real_enc = self.enclitic_pronoun

            else:
                real_pos = self.pos
                real_an1 = self.analysis1
                real_an2 = self.analysis2
                real_an3 = self.analysis3
                real_enc = self.enclitic_pronoun

            for column, val in zip(
                    [POS, ANALYSIS1, ANALYSIS2, ANALYSIS3, ENCLITIC],
                    [real_pos, real_an1, real_an2, real_an3, real_enc]):
                self.set_val(column, val)
        return was_off
    
    def __str__(self):

        return ' '.join([str(self.word), str(self.stem), str(self.pos), str(self.analysis1),
                         str(self.analysis2), str(self.analysis3), str(self.enclitic_pronoun),
                         str(self.comments), str(self.orthography_pronunciation)])


def write_sentences_to_excel(sentences, output_dir):
    columns = ["source", "line", "sentence", "word", "stem", "pos", "analysis1",
               "analysis2", "analysis3", "enclitic pronoun", "comments",
               "orthography and pronunciation"]
    data = {}
    for sentence in sentences:
        source_file = sentence.source[0]
        file_sheets = data.get(source_file, {})
        if source_file not in data:
            data[source_file] = file_sheets
        source_parts = sentence.source[1].split(";")
        sheet_name = source_parts[0]
        sheet_rows = file_sheets.get(sheet_name, [])
        if sheet_name not in file_sheets:
            file_sheets[sheet_name] = sheet_rows
        source = source_parts[1]
        line = source_parts[2]
        raw_sentence = sentence.raw_text
        for word_analysis in sentence.word_analyses:
            word = word_analysis.word
            stem = word_analysis.stem
            pos = word_analysis.pos
            an1 = word_analysis.analysis1
            an2 = word_analysis.analysis2
            an3 = word_analysis.analysis3
            enc = word_analysis.enclitic_pronoun
            comments = word_analysis.comments
            orth_pron = word_analysis.orthography_pronunciation
            row = [source, line, raw_sentence, word, stem, pos, an1, an2, an3,
                               enc, comments, orth_pron]
            row = [val if val != "None" else pd.np.nan for val in row]
            sheet_rows.append(row)
        sheet_rows.append([""]*NUM_COLS)

    for file_name, sheet_names in data.items():
        for sheet_name, sheet_data in sheet_names.items():
            df = pd.DataFrame(sheet_data, columns=columns)
            df.replace('None', '')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # TODO make sure filename does not include extension
            out_path = os.path.join(output_dir, file_name)
            with pd.ExcelWriter(out_path+".xlsx") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
