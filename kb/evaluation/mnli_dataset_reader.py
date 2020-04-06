import json
from typing import Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from allennlp.common.file_utils import cached_path


@DatasetReader.register("multi_nli")
class MultiNLIDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_markers: bool = False):
        super().__init__()
        self.label_to_index = {'contradiction': 0, 'neutral': 1 , 'entailment': 2}
        self.tokenizer = tokenizer_and_candidate_generator
        self.tokenizer.whitespace_tokenize = True
        self.entity_markers = entity_markers

    def text_to_instance(self, line) -> Instance:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterable[Instance]:
        """Creates examples for the training and dev sets."""

        with open(cached_path(file_path), 'r') as fp:
            for line in fp:
                example = json.loads(line)
                text_a = example['sentence1']
                text_b = example['sentence2']
                label = example['gold_label']

                # FIXME: what we do about disagreement on dataset?
                if label == '-':
                    continue

                token_candidates = self.tokenizer.tokenize_and_generate_candidates(text_a, text_b)
                fields = self.tokenizer.convert_tokens_candidates_to_fields(token_candidates)
                fields['label_ids'] = LabelField(self.label_to_index[label], skip_indexing=True)

                instance = Instance(fields)
                yield instance
