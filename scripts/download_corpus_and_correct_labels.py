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
# download_corpus_and_correct_labels.py

"""
This module contains functions to download the CoNLL-2003 corpus and apply label
corrections as done in the paper TODO: PAPER REF.
"""
import argparse
import logging
import os

from correct_label_errors import process_dataset_file

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


def apply_label_corrections(data_set_info, csv_file, target_dir=None, corpus_fold=None):
    """

    :param data_set_info: Dictionary containing a mapping from fold name to file name for
     each of the three folds (`train`, `test`, `dev`) of the corpus.
    :param csv_file: CSV file containing the label corrections
    :param target_dir: (optional) Target directory to for the corrected corpus or
     None for default of "corrected_corpus".
    :param corpus_fold: (optional) Apply corrections to a specific fold only, or None for
     the entire corpus.
    :return:
    """
    target_dir = target_dir or "corrected_corpus"

    fold_n_files = data_set_info.items() if corpus_fold is None \
        else [(corpus_fold, data_set_info[corpus_fold])]

    for fold, fold_file in fold_n_files:
        target_file = os.path.join(target_dir, os.path.split(fold_file)[-1])
        logging.info("Processing fold '{}' to file: '{}'".format(fold, target_file))
        process_dataset_file(fold, fold_file, csv_file, None, target_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="This is a module to help download and apply "
                                                 "label corrections to the CoNLL-2003 corpus.")

    parser.add_argument("--original_corpus_dir", type=str, default="original_corpus",
                        help="Directory for the original CoNLL-2003 corpus")

    parser.add_argument("--corrected_corpus_dir", type=str, default="corrected_corpus",
                        help="Directory to place the corrected corpus")

    parser.add_argument("--corrections_file", type=str,
                        default=os.path.join("corrected_labels",
                                             "all_conll_corrections_combined.csv"))

    parser.add_argument("--corpus_fold", type=str,
                        help="Correct only a specific fold of the corpus if specified as "
                             "[train|dev|test], otherwise with correct the entire corpus")

    args = parser.parse_args()

    # Make sure working dir is top level of project repo
    d = os.getcwd()
    d_split = os.path.split(d)
    if d_split[-1] == "scripts":
        os.chdir(os.path.join(*d_split[:-1]))

    apply_label_corrections(
        get_or_download_corpus(target_dir=args.original_corpus_dir),
        args.corrections_file,
        args.corrected_corpus_dir,
        args.corpus_fold
    )
