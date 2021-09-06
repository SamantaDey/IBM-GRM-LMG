# Log Message Generation: Opportunities
The existing LMG models in today's world are:
- NMT
- NNgen
- LogGen

Our work involves finding ways to improve on the state-of-the-art LogGen model used by CC2Vec towards generating better log messages.

Our lines of work are as follows:
1. Improving the dataset
2. Reduce anisotropy in embedding space of vectors: Incorporating context of code changes in CC2Vec embedded vectors
3. For an unknown code change, summarize top k most similar code changes into a single output log message 
4. Improving the evaluation metric used for assessing quality of test data log messages
