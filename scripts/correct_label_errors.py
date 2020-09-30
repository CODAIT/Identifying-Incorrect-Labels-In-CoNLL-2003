"""
Usage:

    python correct_label_errors.py [train|dev|test] [dataset_file] [csv patch] > new_dataset_file

"""


import math
import re
import sys

import pandas as pd
import text_extensions_for_pandas as tp


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
                correct_line = ' '.join((prefix, "I-" + right_tag))  # TODO: May not be I-
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

    def correct_wrong(self, span, doc_num):
        "Correct a Wrong type error."

        begin_linum, end_linum = self.find(span, doc_num)
        self._correct_wrong_range(begin_linum, end_linum)

    def correct_missing(self, span, right_tag, doc_num):
        "Correct a Missing type error."

        begin_linum, end_linum = self.find(span, doc_num)
        for linum in range(begin_linum, end_linum + 1):
            line = self.dataset_lines[linum]
            prefix, _ = line.rsplit(maxsplit=1)
            correct_line = ' '.join((prefix, f'I-{right_tag}'))  # TODO: This is not necessarily "I-"
            self.dataset_lines[linum] = correct_line

    def correct_span(self, corpus_span, correct_span, doc_num):
        "Correct a Span type error."

        # Get the tag from the corpus span
        corpus_begin_linum, corpus_end_linum = self.find(corpus_span, doc_num)
        _, tag = self.dataset_lines[corpus_begin_linum].rsplit(maxsplit=1)
        if tag != 'O':  # We only want the part after I/B-
            _, tag = self.dataset_lines[corpus_begin_linum].rsplit(sep='-', maxsplit=1)
        else:
            print(f"{corpus_span} has an invalid tag {tag}")

        # correct using the correct span
        begin_linum, end_linum = self.find(correct_span, doc_num)
        # Turn out-of-range lines to O
        self._correct_wrong_range(corpus_begin_linum, begin_linum - 1)
        self._correct_wrong_range(end_linum + 1, corpus_end_linum)
        # In-range lines should be set to the tag of the corpus span
        for linum in range(begin_linum, end_linum + 1):
            line = self.dataset_lines[linum]
            prefix, _ = line.rsplit(maxsplit=1)
            correct_line = ' '.join((prefix, f'I-{tag}'))
            self.dataset_lines[linum] = correct_line
        # TODO: May need to correct examine "I-" to "B-" following this line

    def save(self):
        "Return the corrected dataset file."

        return "\n".join(self.dataset_lines)


def process_dataset_file(dataset_fold, dataset_file, csv_patch_file, csv_encoding=None, target_file=None):
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
                print(f'[WARNING] correct ent type for line {index} are empty. Skipping...',
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
                print(f'[WARNING] Correct_ent_type for line {index} is empty. Skipping...', file=sys.stderr)
                continue
            dataset.correct_tag(row['corpus_span'], row['correct_ent_type'], int(row['doc_offset']))
            dataset.correct_span(row['corpus_span'], row['correct_span'], int(row['doc_offset']))

    result = dataset.save()

    if target_file is not None:
        with open(target_file, mode="w") as f:
            f.writelines(dataset.dataset_lines)

    return result



if __name__ == '__main__':

    dataset_fold = sys.argv[1]
    dataset_file = sys.argv[2]
    csv_patch_file = sys.argv[3]
    csv_encoding = 'latin-1'

    dataset = process_dataset_file(dataset_fold, dataset_file, csv_patch_file, csv_encoding)

    print(dataset.to_string())
