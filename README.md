# Identifying-Incorrect-Labels-In-CoNLL-2003
Research into identifying and correcting incorrect labels in the CoNLL-2003 corpus.

To download the CoNLL-2003 corpus and apply label corrections to produce a corrected version of
the corpus, run the commands below. The CoNLL-2003 corpus is licensed for research use only. Be
sure to adhere to the terms of the license when using this data set!

```bash
pip3 install -r requirements.txt
python3 scripts/download_and_correct_corpus.py
```

This will download the CoNLL-2003 corpus to `original_corpus/`, apply corrections and save the
corrected corpus in `corrected_corpus/`.

NOTE: [Text Extensions for Pandas](https://github.com/CODAIT/text-extensions-for-pandas) must be 
installed to run the script. It provides utilities to download and work with the CoNLL-2003
corpus and assist with NLP analysis on Pandas.
