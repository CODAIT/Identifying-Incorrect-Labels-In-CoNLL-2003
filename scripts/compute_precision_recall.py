
import os
import sys
import logging

import numpy as np
import pandas as pd

# And of course we need the text_extensions_for_pandas library itself.
import text_extensions_for_pandas as tp


def compute_precision_recall(gold_standard, team_output, team_name):

    # Let's stick with exact matches for now.
    # Iterate over the pairs of dataframes for all the documents finding the
    # inputs we need to compute precision and recall for each document, and
    # wrap these values in a new dataframe.
    num_true_positives = [len(gold_standard[i].merge(team_output[i]).index)
                          for i in range(len(gold_standard))]
    num_extracted = [len(df.index) for df in team_output]
    num_entities = [len(df.index) for df in gold_standard]
    doc_num = np.arange(len(gold_standard))

    stats_by_doc = pd.DataFrame({
        "doc_num": doc_num,
        "num_true_positives": num_true_positives,
        "num_extracted": num_extracted,
        "num_entities": num_entities
    })

    # Collection-wide precision and recall can be computed by aggregating
    # our dataframe:
    num_true_positives = stats_by_doc["num_true_positives"].sum()
    num_entities = stats_by_doc["num_entities"].sum()
    num_extracted = stats_by_doc["num_extracted"].sum()

    precision = num_true_positives / num_extracted
    recall = num_true_positives / num_entities
    F1 = 2.0 * (precision * recall) / (precision + recall)

    logging.info("""Team {}:
        Number of correct answers: {}
        Number of entities identified: {}
        Actual number of entities: {}
        Precision: {:1.4f}
        Recall: {:1.4f}
        F1: {:1.4f}""".format(team_name, num_true_positives, num_entities, num_entities, precision,
                              recall, F1))

    return precision, recall, F1


def compute_all(gold_standard_info, team_output_dir, entity_type=None):

    logging.info("Loading gold standard")

    # Read gold standard data for the test set.
    gold_standard = tp.io.conll.conll_2003_to_dataframes(gold_standard_info["test"],
                                                         ["pos", "phrase", "ent"],
                                                         [False, True, True])

    # We don't use the part of speech tags and shallow parse information
    # from the original data set, so strip those columns out.
    gold_standard = [
        df.drop(columns=["pos", "phrase_iob", "phrase_type"])
        for df in gold_standard
    ]

    logging.info("Converting gold standard iob to spans")

    gold_standard_spans = [tp.io.conll.iob_to_spans(df) for df in gold_standard]

    # Load up the results from all 16 teams at once.
    teams = ["bender", "carrerasa", "carrerasb", "chieu", "curran",
             "demeulder", "florian", "hammerton", "hendrickx",
             "klein", "mayfield", "mccallum", "munro", "whitelaw",
             "wu", "zhang"]

    # File with team model test output
    filename = "eng.testb_corrected"

    logging.info("Loading team outputs")

    # Read all the output files into one dataframe per <document, team> pair.
    team_output = {
        t: tp.io.conll.conll_2003_output_to_dataframes(
            gold_standard, os.path.join(team_output_dir, t, filename))
        for t in teams
    }  # Type: Dict[str, List[pd.DataFrame]]

    logging.info("Converting team outputs iob to spans")

    team_output_spans = {t: [tp.io.conll.iob_to_spans(df) for df in dfs]
                         for t, dfs in team_output.items()}

    if entity_type is not None:
        logging.info("Using only {} entity type".format(entity_type))
        gold_standard_spans = [df[df["ent_type"] == entity_type] for df in gold_standard_spans]
        team_output_spans = {t: [df[df["ent_type"] == entity_type] for df in dfs]
                             for t, dfs in team_output_spans.items()}

    logging.info("Computing precision and recall for teams: {}".format(teams))

    team_metrics = {t: compute_precision_recall(gold_standard_spans, team_span, t)
                    for t, team_span in team_output_spans.items()}

    metrics_df = pd.DataFrame.from_records(list(team_metrics.values()), index=team_metrics.keys(),
                                           columns=["Precision", "Recall", "F1"])
    return metrics_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Filenames for the corups
    # NOTE: This data set is licensed for research use only. Be sure to adhere
    #  to the terms of the license when using this data set!
    data_set_info = {
        "train": os.path.join("corrected_corpus", "eng.train_corrected"),
        "dev": os.path.join("corrected_corpus", "eng.testa_corrected"),
        "test": os.path.join("corrected_corpus", "eng.testb_corrected")
    }

    # Directory with team model outputs
    team_output_dir = os.path.join("team_outputs", "corrected_results")

    # Optionally compute for specific entity type or None for all
    entity_type = None  # "PER"

    result = compute_all(data_set_info, team_output_dir, entity_type=entity_type)
    print("\nPrecision/Recall for Entity Type: {}\n\n{}".format(entity_type or "ALL", result))
