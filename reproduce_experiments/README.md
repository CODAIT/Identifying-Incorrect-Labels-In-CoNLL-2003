# Instructions for Reproducing the Experiments

This folder contains the detailed steps that we used to produce the experimental results in Section 7. Before running, ensure you have downloaded and corrected the corpus according to the project home [README.md](../README.md).

## 7.1 Re-evaluation of the Original Competition Entries

To compute precision, recall and F1 scores for the original CoNLL-2003 submitted models on the corrected corpus, run the following command from the project home directory:

        $ python scripts/compute_precision_recall.py
        
The computed scores will be printed to stdout.


## 7.2 Experimental Results on Recent Models

Instructions to reproduce ["Pooled Contextualized Embeddings for Named Entity Recognition"](./akbik2018-2019/README.md).

Instructions to reproduce ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" for NER on CoNLL-03](./devlin2019/README.md).
