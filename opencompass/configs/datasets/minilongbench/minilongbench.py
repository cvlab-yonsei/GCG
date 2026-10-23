from mmengine.config import read_base

with read_base():
    from .minilongbench2wikimqa.minilongbench_2wikimqa_gen import MiniLongBench_2wikimqa_datasets
    from .minilongbenchhotpotqa.minilongbench_hotpotqa_gen import MiniLongBench_hotpotqa_datasets
    from .minilongbenchmusique.minilongbench_musique_gen import MiniLongBench_musique_datasets
    from .minilongbenchmultifieldqa_en.minilongbench_multifieldqa_en_gen import MiniLongBench_multifieldqa_en_datasets
    from .minilongbenchmultifieldqa_zh.minilongbench_multifieldqa_zh_gen import MiniLongBench_multifieldqa_zh_datasets
    from .minilongbenchnarrativeqa.minilongbench_narrativeqa_gen import MiniLongBench_narrativeqa_datasets
    from .minilongbenchqasper.minilongbench_qasper_gen import MiniLongBench_qasper_datasets
    from .minilongbenchtriviaqa.minilongbench_triviaqa_gen import MiniLongBench_triviaqa_datasets
    from .minilongbenchgov_report.minilongbench_gov_report_gen import MiniLongBench_gov_report_datasets
    from .minilongbenchqmsum.minilongbench_qmsum_gen import MiniLongBench_qmsum_datasets
    from .minilongbenchvcsum.minilongbench_vcsum_gen import MiniLongBench_vcsum_datasets
    from .minilongbenchdureader.minilongbench_dureader_gen import MiniLongBench_dureader_datasets
    from .minilongbenchlcc.minilongbench_lcc_gen import MiniLongBench_lcc_datasets
    from .minilongbenchrepobench.minilongbench_repobench_gen import MiniLongBench_repobench_datasets
    from .minilongbenchpassage_retrieval_en.minilongbench_passage_retrieval_en_gen import MiniLongBench_passage_retrieval_en_datasets
    from .minilongbenchpassage_retrieval_zh.minilongbench_passage_retrieval_zh_gen import MiniLongBench_passage_retrieval_zh_datasets
    from .minilongbenchpassage_count.minilongbench_passage_count_gen import MiniLongBench_passage_count_datasets
    from .minilongbenchtrec.minilongbench_trec_gen import MiniLongBench_trec_datasets
    from .minilongbenchlsht.minilongbench_lsht_gen import MiniLongBench_lsht_datasets
    from .minilongbenchmulti_news.minilongbench_multi_news_gen import MiniLongBench_multi_news_datasets
    from .minilongbenchsamsum.minilongbench_samsum_gen import MiniLongBench_samsum_datasets

minilongbench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
