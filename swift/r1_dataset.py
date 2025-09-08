from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset
register_dataset(DatasetMeta(dataset_name='RUC-NLPIR/FlashRAG_datasets', hf_dataset_id='RUC-NLPIR/FlashRAG_datasets', subsets=['nq'], split=['train'], tags=['nq']))
