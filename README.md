EventModel is a 1.2B parameter model finetune of LFM2-1.2B using data extracted from r/parents. The idea is to come up with problems that a kid of certain age would face. This is done by using data from r/Parenting, analyzing the problem, analyzing the kid group using iterative few-shot prompting, then finetunning a generative model with the results.

# TBD: Benchmarking & Analysis
Does this cheap finetune actually perform better than a real model at coming up with some form of reddit post? Certain degree of creativities may need to be done. Perhaps an A/B testing system will be setup to see.