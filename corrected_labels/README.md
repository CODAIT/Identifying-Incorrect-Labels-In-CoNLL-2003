This directory This directory contains corrected labels for the different error types, as well as a single file with
all corrections combined: `all_corrections_combined.csv`. Sub-directories contain the different stages 
of model outputs, human labels and final audited corrections.  

File Name                       | Produced By    | Description
------------------------------- | -------------- | --------------------------------------------------------------------
`all_conll_corrections_combined.csv` | `Label_stats.ipynb` | A consolidated list of all the corrections that we will perform on the corpus
`annotator_rubric.csv` | | A list of owners of annotations/audit of each underlying file
`sentence_corrections.json` | `sentence_correction_preprocessing.ipynb` | Final list of lines to be deleted from the corpus/submissions
`model_outputs` | trained model ensemble outputs | model predictions of correct labels
`human_labels` | Manual inspection of labels | human annotations on top of model outputs |
`human_labels_auditted` | Peer-reviewed audits | secondary review (audit) of above human labels
