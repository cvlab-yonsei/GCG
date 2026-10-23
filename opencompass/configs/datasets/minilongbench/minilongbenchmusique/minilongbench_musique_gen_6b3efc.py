from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, MiniLongBenchDataset

MiniLongBench_musique_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_musique_infer_cfg = dict(
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

MiniLongBench_musique_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

MiniLongBench_musique_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_musique',
        path='./data/MiniLongBench',
        name='musique',
        reader_cfg=MiniLongBench_musique_reader_cfg,
        infer_cfg=MiniLongBench_musique_infer_cfg,
        eval_cfg=MiniLongBench_musique_eval_cfg,
    )
]
