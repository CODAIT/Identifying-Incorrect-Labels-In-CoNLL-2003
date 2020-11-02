# data directory

Data for hand-labeling.

Please keep this table up to date as you add files. We may need to refer back
to this directory several months down the road.


File Name                       | Produced By    | Description
------------------------------- | -------------- | --------------------------------------------------------------------
`CoNLL_2_in_gold.csv`           | `CoNLL_2.ipynb`| Entities in gold standard but not in competitors' outputs (test folds)
`CoNLL_2_in_gold_BC.csv`        | Bryan          | Labels for `CoNLL_2_in_gold.csv`
`CoNLL_2_not_in_gold.csv`       | `CoNLL_2.ipynb`| Entities in competitors' outputs but not in gold standard (test folds)
`CoNLL_2_not_in_gold_FRR.csv`   | Fred           | Labels for `CoNLL_2_not_in_gold.csv`
`CoNLL_3_in_gold.csv`           | `CoNLL_3.ipynb`| Entities in gold standard but not in model outputs (test folds)
`CoNLL_3_not_in_gold.csv`       | `CoNLL_3.ipynb`| Entities in model outputs but not in gold standard (test folds)
`CoNLL_3_train_in_gold.csv`     | `CoNLL_3.ipynb`| Entities in gold standard but not in model outputs (train fold)
`CoNLL_3_train_not_in_gold.csv` | `CoNLL_3.ipynb`| Entities in model outputs but not in gold standard (train fold)
`CoNLL_4_in_gold.csv`           | `CoNLL_4.ipynb`| Entities in gold standard but not in model outputs (test folds)
`CoNLL_4_not_in_gold.csv`       | `CoNLL_4.ipynb`| Entities in model outputs but not in gold standard (test folds)
`CoNLL_4_train_in_gold.csv`     | `CoNLL_4.ipynb`| Entities in gold standard but not in model outputs (train fold)
`CoNLL_4_train_not_in_gold.csv` | `CoNLL_4.ipynb`| Entities in model outputs but not in gold standard (train fold)
`sentence_correction_xxx.csv` | `sentence_correction_preprocessing.ipynb` | Intermediate CSV files for sentence correction
`sentence_corrections.json` | `sentence_correction_preprocessing.ipynb` | Final list of lines to be deleted from the corpus/submissions
`all_conll_corrections_combined.csv` | `Label_stats.ipynb` | A consolidated list of all the corrections that we will perform on the corpus
`modified_results/` | `apply_sentence_corrections.ipynb` | This folder contains the 16 team submissions and their modified submissions after applying sentence corrections to each submission. The modified files are tagged with a `_corrected` suffix.  




