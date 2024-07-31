# Open Domain Question Answering
RAFT, TAPT, Instruction Tuning, and Preference Optimization on [google/gemma-2b](https://huggingface.co/google/gemma-2b) for Open Domain Question Answering
## Performance
### Hyperparameters
| Hp | Retrieval | Reranker | Reader | 
| ------------- | ------------- | ------------- | ------------- |
| lr |  |  |  |
| bs |  |  |  |
| scheduler |  |  |  |
| gpu | rtx 4090  | rtx 4090  | rtx 4090  |
| epoch |  |  |  |
| input maxlength |  |  |  |
| batch size |   |  |  |
| hard neg | 8 |  |  |
| τ | 0.02 |  |  |
### Retrieval
| Method | R-Precision | Recall@5 | TPS | 
| ------------- | ------------- | ------------- | ------------- |
| BM25 | 67.018 | 91.265 | 0.007s |
| Dense |  73.343 | 95.03 | 0.44s |
| Cross Encoder | 86.747 | 96.084 | 1.30s |

### Reader
| Method | EM | F-1 | Rouge-L |
| ------------- | ------------- | ------------- |------------- |
| SFT (Close-book) | 0  | 2.959  | 10.614 |
| SFT (Open-book) | 0 | 3.423  | 77.136 |
| Instruction Tuning | 0 |  3.4000 | 89.71 | 
| Instruction Tuning + Synthetic SFT | 0  | 3.404  | 90.426 |
| Instruction Tuning + Synthetic SFT + SFT |  0  | 3.388  | 90.305 |
| Instruction Tuning + Synthetic SFT + SimPO  |  0 | 3.407  | 90.746| 


## Directory Structure
```
├── README.md
├── refine.py #perform data cleaning
└── requirements.txt
```

## Install Required Packages
```pip install -r requirements.txt```

## Data Preprocessing
### Clean Data
```
python src/clean/clean.py # html parsing
python src/clean/map.py # map negative pairs and filter data with more than 1 gold table for Retrieval
python src/clean/negatives.py # clean corpus and map rejection samples for Preference Optmization
```
### Train/Test split
```
bash src/split/split.sh
```

### Synthetic Data Generation
```
python src/synthetic/question_generation.py # question generation
python src/synthetic/answer_generation.py #answer generation
python src/synthetic/refine_generation.py #self-refine generation
python src/synthetic/filter.py #consistent-based filtering (sort by l2 distance to the train label)
```

## Retrieval
### Data Preparation
```python src/clean/copy_train.py```
### Training
#### Dense Retrieval
```bash src/retrieval/dense_finetuning.sh```
#### Cross Encoder Reranker
```bash src/retrieval/ce_finetuning.sh```
### Inferenece
```python src/retrieval/retrieval.py```



## SFT

## Instruction Tuning

## Simple Preference Optimization
