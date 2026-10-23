from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, MiniLongBenchDataset

MiniLongBench_multifieldqa_en_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_multifieldqa_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

MiniLongBench_multifieldqa_en_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

MiniLongBench_multifieldqa_en_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_multifieldqa_en',
        path='./data/MiniLongBench',
        name='multifieldqa_en',
        reader_cfg=MiniLongBench_multifieldqa_en_reader_cfg,
        infer_cfg=MiniLongBench_multifieldqa_en_infer_cfg,
        eval_cfg=MiniLongBench_multifieldqa_en_eval_cfg,
    )
]
