from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, MiniLongBenchDataset

MiniLongBench_2wikimqa_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_2wikimqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32),
)

MiniLongBench_2wikimqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

MiniLongBench_2wikimqa_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_2wikimqa',
        path='./data/MiniLongBench',
        name='2wikimqa',
        reader_cfg=MiniLongBench_2wikimqa_reader_cfg,
        infer_cfg=MiniLongBench_2wikimqa_infer_cfg,
        eval_cfg=MiniLongBench_2wikimqa_eval_cfg,
    )
]
