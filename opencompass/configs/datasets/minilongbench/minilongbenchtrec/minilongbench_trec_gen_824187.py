from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchClassificationEvaluator,
    MiniLongBenchDataset,
    trec_postprocess,
)

MiniLongBench_trec_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='all_labels',
    train_split='test',
    test_split='test',
)

MiniLongBench_trec_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

MiniLongBench_trec_eval_cfg = dict(
    evaluator=dict(type=LongBenchClassificationEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=trec_postprocess),
)

MiniLongBench_trec_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_trec',
        path='./data/MiniLongBench',
        name='trec',
        reader_cfg=MiniLongBench_trec_reader_cfg,
        infer_cfg=MiniLongBench_trec_infer_cfg,
        eval_cfg=MiniLongBench_trec_eval_cfg,
    )
]
