# Reproduce "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" for NER on CoNLL-03

1. Create a new conda env and install pytorch 1.5.1:

       $ conda create -n bert python=3.7
       $ conda activate bert
       $ conda install pytorch=1.5.1 cudatoolkit=10.2 -c pytorch

1. Clone transformer, check out version v2.11.0:

       $ git clone https://github.com/huggingface/transformers
       $ cd transformers
       $ git checkout v2.11.0

1. Install from source by following [the instructions](https://github.com/huggingface/transformers#from-source).

1. Apply a relevant bug-fix patch to the NER script:

       $ wget https://patch-diff.githubusercontent.com/raw/huggingface/transformers/pull/5326.patch
       $ git apply 5326.patch

1. Copy CoNLL-03 data files to `examples/token-classification/conll03`. There should be three files in `conll03`: `eng.train`, `eng.testa`, and `eng.testb`. Rename them to `train.txt`, `dev.txt`, and `test.txt`, respectively.

1. Save the following to `examples/token-classification/config.json`:

   ```json
   {
    "data_dir": "./conll03",
    "model_name_or_path": "bert-large-cased",
    "output_dir": "conll03-model",
    "max_seq_length": 512,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "save_steps": 750,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true
   }
   ```

   Note that the paper does not specify the exact number of epochs and batch size. In section A.3, it lists a few suggestions.

1. Install `seqeval`:

       $ pip install seqeval[cpu]

1. Run the NER script:

       $ cd examples/token-classification
       $ CUDA_VISIBLE_DEVICES="" nice python run_ner.py config.json |& tee out.txt
