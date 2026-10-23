from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchF1Evaluator,
    MiniLongBenchDataset,
    triviaqa_postprocess,
)

MiniLongBench_triviaqa_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32),
)

MiniLongBench_triviaqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=triviaqa_postprocess),
)

MiniLongBench_triviaqa_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_triviaqa',
        path='./data/MiniLongBench',
        name='triviaqa',
        reader_cfg=MiniLongBench_triviaqa_reader_cfg,
        infer_cfg=MiniLongBench_triviaqa_infer_cfg,
        eval_cfg=MiniLongBench_triviaqa_eval_cfg,
    )
]
