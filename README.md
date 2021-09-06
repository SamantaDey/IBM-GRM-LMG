# Comapring standard MT Evaluation Metrics
Good evaluation metrics for LMG should:
1. Have high correlation with human judgements
2. Capture the semantics of log messages well

**The popular BLEU metric, used by the state-of-art LMG models, fails to capture log message semantics**

| Given Message | Predicted Message | BLEU-4 score |
| ------------- | ----------------- |------------- |
| Remove bogus symlink .  | Remove mistake symlink .  | 0.1073 |
| add Yan to team  | added person  | 0 |
| changed path to libraries | Updated library paths | 0 |


Here, we understand the working of some standard Automatic MT Evaluation Metrics, which could be possibly used for LMG Evaluation

We consider three BLEU variants:
- BLEU-Moses
- BLEU-Norm
- BLEU-CC

Apart from this, we also compare other standard MT metrics, like:
* NIST
* METEOR
* ROUGE

