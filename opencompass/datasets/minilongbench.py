import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset

_CLASSIFICATION_TASKS = {'lsht', 'trec'}
_NO_INPUT_TASKS = {'lcc'}


@LOAD_DATASET.register_module()
class MiniLongBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        jsonl = os.path.join(path, 'data', f'{name}.jsonl')
        items = []
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

        raw = []
        for item in items:
            context = item['context']
            answers = item['answers']
            question = item.get('input', '')
            all_classes = item.get('all_classes', None)
            if name in _CLASSIFICATION_TASKS:
                raw.append({
                    'input': question,
                    'context': context,
                    'all_labels': {
                        'answers': answers,
                        'all_classes': all_classes,
                    },
                })
            elif name in _NO_INPUT_TASKS:
                raw.append({'context': context, 'answers': answers})
            else:
                raw.append({
                    'input': question,
                    'context': context,
                    'answers': answers,
                })

        return DatasetDict({'test': Dataset.from_list(raw)})


@TEXT_POSTPROCESSORS.register_module()
def minilongbench_lsht_postprocess(text: str) -> str:
    return text.lstrip('\n').split('\n')[0]


@TEXT_POSTPROCESSORS.register_module()
def minilongbench_trec_postprocess(text: str) -> str:
    return text.lstrip('\n').split('\n')[0]
