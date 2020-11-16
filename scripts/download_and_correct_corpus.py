#
#  Copyright (c) 2020 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

################################################################################
# download_and_correct_corpus.py

"""
This module contains functions to download the CoNLL-2003 corpus then applies
label and sentence boundary corrections to create a corrected corpus.
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile

import pandas as pd

try:
    import text_extensions_for_pandas as tp
except ImportError as e:
    raise ImportError("Text Extensions for Pandas is required to run this script. Please "
                      "install with `pip install text-extensions-for-pandas` or see the "
                      "project page at https://github.com/CODAIT/text-extensions-for-pandas "
                      "for more information.\ncaused by\n{}".format(str(e)))


def get_or_download_corpus(target_dir=None):
    """
    Download the CoNLL-2003 corpus or load a previously downloaded copy.

    NOTE: This data set is licensed for research use only. Be sure to adhere
      to the terms of the license when using this data set!
    :param target_dir: (optional) Target directory to download the corpus or
     None for default of "original_corpus".
    :return: Dictionary containing a mapping from fold name to file name for
     each of the three folds (`train`, `test`, `dev`) of the corpus.
    """
    # Download and cache the data set.
    # NOTE: This data set is licensed for research use only. Be sure to adhere
    #  to the terms of the license when using this data set!
    target_dir = target_dir or "original_corpus"
    logging.info("Getting CoNLL-2003 Corpus..")
    data_set_info = tp.io.conll.maybe_download_conll_data(target_dir)
    logging.info("CoNLL-2003 Corpus downloaded to: {}"
                 .format(os.path.join(os.getcwd(), target_dir)))
    return data_set_info


class Dataset:
    _span_pattern = re.compile(r'\[(\d+),\s*(\d+)\)')

    def __init__(self, dataset_file):
        with open(dataset_file) as f:
            self.dataset_lines = f.read().splitlines()

        self.dataset_dfs = tp.io.conll.conll_2003_to_dataframes(
            dataset_file, ["pos", "phrase", "ent"], [False, True, True])

    @classmethod
    def _extract_begin_end_from_span(cls, span):
        "Extract the begin and end from a given span. Returns (begin, end)."

        m = cls._span_pattern.match(span)
        if m is None:
            raise ValueError(f'Invalid span: {span}')
        return int(m.group(1)), int(m.group(2))

    def find(self, span, doc_num):
        "Find the line number of a given span."

        df = self.dataset_dfs[doc_num]
        try:
            begin, end = self._extract_begin_end_from_span(span)
        except ValueError:
            print(f"[WARNING] Invalid span {span}", file=sys.stderr)
            return -1, -2
        found_df = df[df["span"].values.begin == begin]
        if found_df.shape[0] == 0:
            print(f"[WARNING] Could not find {span}", file=sys.stderr)
            return -1, -2
        begin_linum = found_df.iloc[0]['line_num']
        found_df = df[df["span"].values.end == end]
        if found_df.shape[0] == 0:
            print(f"[WARNING] Could not find {span}", file=sys.stderr)
            return -1, -2
        end_linum = found_df.iloc[0]['line_num']
        return begin_linum, end_linum

    def correct_tag(self, span, right_tag, doc_num):
        "Correct tag of a given span."

        begin_linum, end_linum = self.find(span, doc_num)
        for linum in range(begin_linum, end_linum + 1):
            line = self.dataset_lines[linum]
            prefix, tag = line.rsplit(maxsplit=1)
            if tag == 'O':
                # Determine type
                if linum == begin_linum and linum != 0:
                    type_ = self._determine_type(self.dataset_lines[linum - 1], right_tag)
                else:
                    type_ = "I"

                correct_line = ' '.join((prefix, f"{type_}-" + right_tag))
            else:
                prefix, _ = line.rsplit(sep='-', maxsplit=1)
                correct_line = '-'.join((prefix, right_tag))
            self.dataset_lines[linum] = correct_line

    def _correct_wrong_range(self, begin_linum, end_linum):
        "Correct Wrong type error given a line range."

        for linum in range(begin_linum, end_linum + 1):
            line = self.dataset_lines[linum]
            if not line.strip():  # Skip blank lines
                continue
            prefix, _ = line.rsplit(maxsplit=1)
            correct_line = ' '.join((prefix, 'O'))
            # TODO: Need to examine the correctness of nearby "B-" and "I-"
            self.dataset_lines[linum] = correct_line

        # If the line below the corrected range is "B-", it should be changed to "I-"
        if end_linum < len(self.dataset_lines) - 1:  # end_linum is not the last line
            self.dataset_lines[end_linum + 1] = self._correct_line_i_b(self.dataset_lines[end_linum],
                                                                       self.dataset_lines[end_linum + 1])

    def correct_wrong(self, span, doc_num):
        "Correct a Wrong type error."

        begin_linum, end_linum = self.find(span, doc_num)
        self._correct_wrong_range(begin_linum, end_linum)

    def correct_missing(self, span, right_tag, doc_num):
        "Correct a Missing type error."

        begin_linum, end_linum = self.find(span, doc_num)
        for linum in range(begin_linum, end_linum + 1):
            # Determine type
            if linum == begin_linum and linum != 0:
                type_ = self._determine_type(self.dataset_lines[linum - 1], right_tag)
            else:
                type_ = "I"

            line = self.dataset_lines[linum]
            prefix, _ = line.rsplit(maxsplit=1)
            correct_line = ' '.join((prefix, f'{type_}-{right_tag}'))
            self.dataset_lines[linum] = correct_line

    def correct_span(self, corpus_span, correct_span, doc_num):
        "Correct a Span type error."

        # Get the tag from the corpus span
        corpus_begin_linum, corpus_end_linum = self.find(corpus_span, doc_num)
        _, tag = self.dataset_lines[corpus_begin_linum].rsplit(maxsplit=1)
        if tag != 'O':  # We only want the part after I/B-
            _, tag = self.dataset_lines[corpus_begin_linum].rsplit(sep='-', maxsplit=1)
        else:
            print(f"{corpus_span} has an invalid tag {tag}", file=sys.stderr)

        # correct using the correct span
        begin_linum, end_linum = self.find(correct_span, doc_num)
        # Turn out-of-range lines to O
        self._correct_wrong_range(corpus_begin_linum, begin_linum - 1)
        self._correct_wrong_range(end_linum + 1, corpus_end_linum)
        # In-range lines should be set to the tag of the corpus span
        for linum in range(begin_linum, end_linum + 1):
            line = self.dataset_lines[linum]
            prefix, _ = line.rsplit(maxsplit=1)

            # Determine type
            if linum == begin_linum and linum != 0:
                type_ = self._determine_type(self.dataset_lines[linum - 1], tag)
            else:
                type_ = "I"

            correct_line = ' '.join((prefix, f'{type_}-{tag}'))
            self.dataset_lines[linum] = correct_line

        # Next line type may need correction
        if end_linum < len(self.dataset_lines) - 1:  # not last line
            self.dataset_lines[end_linum + 1] = self._correct_line_i_b(self.dataset_lines[end_linum],
                                                                       self.dataset_lines[end_linum + 1])

    def _determine_type(self, prev_line, current_line_tag):
        '''Determine whether the current line should be "I-" or "B-". ``current_line_tag`` is the part after "I-" or
        "B-", or "O".
        '''

        type_and_tag = prev_line.rsplit(maxsplit=1)
        if (len(type_and_tag) == 2 and type_and_tag[1].startswith(("I-", "B-")) and
            type_and_tag[1].endswith(f"-{current_line_tag}")):
            # previous line is I or B type and has the same tag
            return "B"

        return "I"

    def _correct_line_i_b(self, prev_line, current_line):
        "Correct the I- and B- type of the current line."

        prefix_and_tag = current_line.rsplit(maxsplit=1)
        if len(prefix_and_tag) <= 1:  # blank line
            return current_line
        tag = prefix_and_tag[1]
        if tag == 'O':  # no I- or B- distinction
            return current_line
        tag = tag.rsplit('-', maxsplit=1)[1]
        type_  = self._determine_type(prev_line, tag)
        return ' '.join((prefix_and_tag[0], f'{type_}-{tag}'))

    def save(self):
        "Return the corrected dataset file."

        return "\n".join(self.dataset_lines)


def process_label_file(dataset_fold, dataset_file, csv_patch_file, csv_encoding=None, target_file=None):
    csv_patch = pd.read_csv(csv_patch_file, encoding=csv_encoding)

    # Only look into relevant dataset type rows
    csv_patch = csv_patch[csv_patch['fold'] == dataset_fold]
    # Skip all None errors
    csv_patch = csv_patch[csv_patch['error_type'] != 'None']

    dataset = Dataset(dataset_file)
    for index, row in csv_patch.iterrows():
        # A lot of rows misplace correct_span and corpus_span. If corpus_span
        # is empty, use correct_span as the corpus_span for Missing, Tag, and Wrong.
        corpus_span = row['corpus_span'] if isinstance(row['corpus_span'], str) else row['correct_span']

        if isinstance(corpus_span, float) and math.isnan(corpus_span):
            print(f'[WARNING] Both corpus span and correct span for line {index} are empty. Skipping...',
                  file=sys.stderr)
            continue

        if row['error_type'] == 'Missing':
            if row['correct_span'] == "[53,64) 'West Indes'":
                print(f"Skip span error for {row['correct_span']}. Please correct it by hand.", file=sys.stderr)
                continue
            if isinstance(row['correct_ent_type'], float) and math.isnan(row['correct_ent_type']):
                print(f'[WARNING] correct ent type for line {index} are empty. row: {row}. Skipping...',
                      file=sys.stderr)
                continue
            dataset.correct_missing(corpus_span, row['correct_ent_type'], int(row['doc_offset']))
            continue
        elif row['error_type'] == 'Tag':
            dataset.correct_tag(corpus_span, row['correct_ent_type'], int(row['doc_offset']))
        elif row['error_type'] == 'Wrong':
            dataset.correct_wrong(corpus_span, int(row['doc_offset']))

    for index, row in csv_patch.iterrows():
        if row['error_type'] == 'Span':
            if row['corpus_span'].endswith("'Minn'"):
                print("Skip span error for '(Iowa-S) Minn'. Please correct it by hand.", file=sys.stderr)
                continue
            if row['corpus_span'].endswith("'Boxing-Bruno'"):
                print(f"Skip span error for '{row['corpus_span']}'. Please correct it by hand.", file=sys.stderr)
                continue
            if row['correct_span'] == "[43, 47): 'U.N.'":
                print(f"Skip span error for '{row['correct_span']}'. Please correct it by hand.", file=sys.stderr)
                continue
            if isinstance(row['correct_span'], float) and math.isnan(row['correct_span']):
                print(f'[WARNING] Correct span for line {index} is empty. Skipping...', file=sys.stderr)
                continue
            dataset.correct_span(row['corpus_span'], row['correct_span'], int(row['doc_offset']))
        elif row['error_type'] == 'Both':
            if isinstance(row['correct_ent_type'], float) and math.isnan(row['correct_ent_type']):
                print(f'[WARNING] Correct_ent_type for line {index} is empty. row: {row}. Skipping...', file=sys.stderr)
                continue
            dataset.correct_tag(row['corpus_span'], row['correct_ent_type'], int(row['doc_offset']))
            dataset.correct_span(row['corpus_span'], row['correct_span'], int(row['doc_offset']))

    result = dataset.save()

    with open(target_file, mode="w") as f:
        f.write(result)

    return result


def process_sentence_file(dataset_fold, dataset_file, json_file, target_file):
    with open(json_file) as f:
        lines_to_delete = json.load(f)

    with open(dataset_file, "r") as source_file:
        file_lines = source_file.readlines()
    for l in sorted(lines_to_delete[dataset_fold], reverse=True):
        if file_lines[l] != "\n":
            raise ValueError("Not deleting a blank line: {}".format(file_lines[l]))
        del file_lines[l]
    with open(target_file, "w+") as new_file:
        for l in file_lines:
            new_file.write(l)


def process_token_file(dataset_fold, dataset_file, sentence_json_file, token_edits_json_file, target_file):
    with open(token_edits_json_file) as f:
        edits = pd.read_json(f)
    edits = edits[edits.fold == dataset_fold]  # select only correct fold
    with open(sentence_json_file) as f:
        sentence_deletes = json.load(f)
    with open(dataset_file, "r") as source_file:
        file_lines = source_file.readlines()

    removed = 0
    for l in range(0, edits.index.max()):
        if l in sentence_deletes[dataset_fold]:
            removed += 1
        if l in edits.index:
            file_lines[l-removed] = edits.at[l, 'correct_line']
    with open(target_file, "w+") as new_file:
        for l in file_lines:
            new_file.write(l)


def apply_corrections(data_set_info, label_csv_file, sentence_json_file, token_edits_json_file, target_dir=None, corpus_fold=None):
    """
    Applies label and sentence boundary corrections
    :param data_set_info: Dictionary containing a mapping from fold name to file name for
     each of the three folds (`train`, `test`, `dev`) of the corpus.
    :param label_csv_file: CSV file containing the label corrections
    :param sentence_json_file: JSON file containing the sentence boundary corrections -- specifically the line numbers
     in each file to be deleted
    :param token_edits_json_file: JSON file containing token edit corrections.
    :param target_dir: (optional) Target directory for the corrected corpus or
     None for default of "corrected_corpus".
    :param corpus_fold: (optional) Apply corrections to a specific fold only, or None for
     the entire corpus.
    """
    target_dir = target_dir or "corrected_corpus"

    fold_n_files = data_set_info.items() if corpus_fold is None \
        else [(corpus_fold, data_set_info[corpus_fold])]

    for fold, fold_file in fold_n_files:
        target_file = os.path.join(target_dir, os.path.split(fold_file)[-1])

        with tempfile.NamedTemporaryFile() as temp_file:
            logging.info("Correcting labels for fold '{}'".format(fold))
            process_label_file(fold, fold_file, label_csv_file, None, temp_file.name)

            logging.info("Correcting sentence boundaries for fold '{}'".format(fold))
            process_sentence_file(fold, temp_file.name, sentence_json_file, temp_file.name)

            logging.info("Correcting token errors for fold '{}'".format(fold))
            process_token_file(fold,temp_file.name,sentence_json_file, token_edits_json_file,target_file)

        logging.info("Corrected corpus fold '{}' to file: '{}'".format(fold, target_file))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="This is a module to help download and apply "
                                                 "label corrections to the CoNLL-2003 corpus.")

    parser.add_argument("--original_corpus_dir", type=str, default="original_corpus",
                        help="Directory for the original CoNLL-2003 corpus")

    parser.add_argument("--corrected_corpus_dir", type=str, default="corrected_corpus",
                        help="Directory to place the corrected corpus")

    parser.add_argument("--label_corrections_file", type=str,
                        default=os.path.join("corrected_labels",
                                             "all_conll_corrections_combined.csv"))

    parser.add_argument("--sentence_boundary_corrections_file", type=str,
                        default=os.path.join("corrected_labels",
                                             "sentence_corrections.json"))

    parser.add_argument("--token_corrections_file", type=str,
                        default=os.path.join("corrected_labels",
                                             "token_corrections.json"))

    parser.add_argument("--corpus_fold", type=str,
                        help="Correct only a specific fold of the corpus if specified as "
                             "[train|dev|test], otherwise with correct the entire corpus")

    args = parser.parse_args()

    # Make sure working dir is top level of project repo
    d = os.getcwd()
    d_split = os.path.split(d)
    if d_split[-1] == "scripts":
        os.chdir(os.path.join(*d_split[:-1]))

    # Verify expected dirs and files exist
    req_paths = {
        "original_corpus_dir": args.original_corpus_dir,
        "corrected_corpus_dir": args.original_corpus_dir,
        "label_corrections_file": args.label_corrections_file,
        "token_corrections_file": args.token_corrections_file,
        "sentence_boundary_corrections_file": args.sentence_boundary_corrections_file,
    }

    missing = {k: v for k, v in req_paths.items() if not os.path.exists(v)}
    if len(missing) > 0:
        raise RuntimeError("Could not find the following files/directories, please verify script "
                           "is run under the project root directory:\n{}".format(missing))

    apply_corrections(
        get_or_download_corpus(target_dir=args.original_corpus_dir),
        args.label_corrections_file,
        args.sentence_boundary_corrections_file,
        args.token_corrections_file,
        args.corrected_corpus_dir,
        args.corpus_fold
    )
