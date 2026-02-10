# EventModel 1.2B

HF Link: [https://huggingface.co/mzen/EventModel-1.2B]

EventModel is a 1.2B parameter model finetune of LFM2-1.2B using data extracted from r/parents. The idea is to come up with problems that a kid of certain age would face. This is done by using data from r/Parenting, analyzing the problem, analyzing the kid group using iterative few-shot prompting, then finetunning a generative model with the results.

<img width="1953" height="258" alt="image" src="https://github.com/user-attachments/assets/0bb0b0c5-62fe-4043-9c23-cdf04df045b1" />

## Running Locally
```
from transformers import pipeline

pipe = pipeline(
    "text-generation", 
    model="mzen/EventModel-1.2B", 
    trust_remote_code=True,
    device_map="auto"
)

prompt = "### Character: 13 year old, boy\n\n### Problem:"

output = pipe(
    prompt,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False 
)

print(output[0]['generated_text'])
```

## TBD: Benchmarking & Analysis
Does this cheap finetune actually perform better than the base model (and/or bigger models) at coming up with some form of interesting problems? Certain degree of creativities may need to be done. Perhaps an A/B testing system will be setup to see.
