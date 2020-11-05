# data directory

Data for hand-labeling.

File Name                       | Produced By    | Description
------------------------------- | -------------- | --------------------------------------------------------------------
`all_conll_corrections_combined.csv` | `Label_stats.ipynb` | A consolidated list of all the corrections that we will perform on the corpus
`sentence_corrections.json` | `sentence_correction_preprocessing.ipynb` | Final list of lines to be deleted from the corpus/submissions
`model_outputs` | original model outputs |
`human_labels` | human annotations on top of model outputs |
`human_labels_auditted` | secondary review (audit) of above human labels
`Inter Annotator Agreement.ipynb` | A notebook analysing the relations between different stages of the correction process
`annotator_rubric.csv` | A list of owners of annotations/audit of each underlying file
