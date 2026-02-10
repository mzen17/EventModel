=== ChildModel
A project for modeling child problem agencies in LLMs.

=== Procedure


We filtered out the is_self tag, reducing our dataset from 587,969 posts to 561,287. This tag controls for if the post is external or not. 
1. is_self tag filter
2. >300 character filter.

1. Check the control on 300 word filter.



We started with manual annotations of [3] data points. Using these as few-shots, we then computed 500 answers of a random sample using Qwen3-VL to run through the data, identifying the problem and character identities as tags.