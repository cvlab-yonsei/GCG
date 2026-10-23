from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, MiniLongBenchDataset

MiniLongBench_qmsum_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

MiniLongBench_qmsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

MiniLongBench_qmsum_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator), pred_role='BOT'
)

MiniLongBench_qmsum_datasets = [
    dict(
        type=MiniLongBenchDataset,
        abbr='MiniLongBench_qmsum',
        path='./data/MiniLongBench',
        name='qmsum',
        reader_cfg=MiniLongBench_qmsum_reader_cfg,
        infer_cfg=MiniLongBench_qmsum_infer_cfg,
        eval_cfg=MiniLongBench_qmsum_eval_cfg,
    )
]
