from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchClassificationEvaluator,
    MiniLongBenchDataset,
    lsht_postprocess,
)

MiniLongBench_lsht_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='all_labels',
    train_split='test',
    test_split='test',
)

MiniLongBench_lsht_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

MiniLongBench_lsht_eval_cfg = dict(
    evaluator=dict(type=LongBenchClassificationEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=lsht_postprocess),
)

MiniLongBench_lsht_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_lsht',
        path='./data/MiniLongBench',
        name='lsht',
        reader_cfg=MiniLongBench_lsht_reader_cfg,
        infer_cfg=MiniLongBench_lsht_infer_cfg,
        eval_cfg=MiniLongBench_lsht_eval_cfg,
    )
]
