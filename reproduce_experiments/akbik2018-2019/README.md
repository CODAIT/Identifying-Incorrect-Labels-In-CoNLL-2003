# Reproduce "Pooled Contextualized Embeddings for Named Entity Recognition"

1. Create directory `resources/tasks/conll_03` and put the three dataset files (`eng.train`, `eng.testa`, and `eng.testb`) in it.

2. Install flair, Python 3.7, PyTorch 1.4.0 (PyTorch 1.5.0 has a bug that prevents the reproduction process from succeeding https://github.com/pytorch/pytorch/issues/37703 ):

       $ pip install torch==1.4.0 flair==1.5.0

3. Run the training script:

       $ python train_akbik2019.py  # or train_akbik2018.py if it is for the older model.

The resulting model will be in `resource/tagger`.

4. Run the evaluation script:

       $ python eval_akbik2019.py  # or train_akbik2018.py if it is for the older model.

   Since it's pretty resource intensive, to prevent the server from becoming too busy and locked down, you may want to limit RAM usage by using `ulimit -v`, lower and use CPU instead of GPU. For example:

       $ ulimit -v 99000000000
       $ CUDA_VISIBLE_DEVICES="" nice python eval.py

Full citation to the paper:

Pooled Contextualized Embeddings for Named Entity Recognition. Alan Akbik, Tanja Bergmann and Roland Vollgraf. 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics, NAACL 2019.

Contextualized Embeddings for Named Entity Recognition. Alan Akbik, Tanja Bergmann and Roland Vollgraf. 2018 International Conference on Computational Linguistics, CoNLL 2018.



Code reproduction reference (akbik2019): https://github.com/flairNLP/flair/blob/master/resources/docs/EXPERIMENTS.md#conll-03-named-entity-recognition-english
