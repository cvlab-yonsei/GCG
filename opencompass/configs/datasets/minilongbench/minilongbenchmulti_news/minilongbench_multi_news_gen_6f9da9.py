from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, MiniLongBenchDataset

MiniLongBench_multi_news_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_multi_news_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:\n',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

MiniLongBench_multi_news_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator), pred_role='BOT'
)

MiniLongBench_multi_news_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_multi_news',
        path='./data/MiniLongBench',
        name='multi_news',
        reader_cfg=MiniLongBench_multi_news_reader_cfg,
        infer_cfg=MiniLongBench_multi_news_infer_cfg,
        eval_cfg=MiniLongBench_multi_news_eval_cfg,
    )
]
