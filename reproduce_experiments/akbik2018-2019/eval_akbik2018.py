from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.models import SequenceTagger

# Load the corpus
corpus: Corpus = CONLL_03(base_path='resources/tasks')

tagger: SequenceTagger = SequenceTagger.load('resources/taggers/akbik2018-ner/final-model.pt')

# Uncomment the following to use the author provided tagger (model)
# tagger: SequenceTagger = SequenceTagger.load('ner')

result, _ = tagger.evaluate([corpus.test])

print(result.detailed_results)
