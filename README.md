EventModel is a 1.2B parameter model finetune of LFM2-1.2B using data extracted from r/parents. The idea is to come up with problems that a kid of certain age would face. This is done by using data from r/Parenting, analyzing the problem, analyzing the kid group using iterative few-shot prompting, then finetunning a generative model with the results.

<img width="1953" height="258" alt="image" src="https://github.com/user-attachments/assets/0bb0b0c5-62fe-4043-9c23-cdf04df045b1" />

# TBD: Benchmarking & Analysis
Does this cheap finetune actually perform better than the base model (and/or bigger models) at coming up with some form of interesting problems? Certain degree of creativities may need to be done. Perhaps an A/B testing system will be setup to see.
