# Analysis

## Layer 4, Head 6
It seems that this attention head is doing the opposite of what Layer 3 head 10 is doing: each token is paying attention to the token before it.


Example Sentences:
- we [MASK] there are a lot of abundance.
- someday we are going to rise up on that [MASK].

## Layer 5, Head 7
It seems that this attention head is distributes attention in such a way that tokens are paying attention to relatable groups of tokens before it. In the sentence "he looks back to the 60s with [MASK]" [MASK] token pays attention to the two groups of tokens: "looks", "back" and "60s", "with", which is might somehow be related to how it tries to describe the sense of a sentence?


Example Sentences:
- [MASK] turned out that everything worked out.
- he looks back to the 60s with [MASK] 


## Additional discoveries:  
Example sentence:
- someday we are going to rise up on that [MASK].  

l4h10 token rise pays attention to tokens up, on
l4h12 each token pays attention to several tokens before it, that it relates to
l5h7 tokens we, are, going, to pay attention to "someday"
l6h9 tokens we, are, going, to pay attention to  "rise"
l8h1 tokens to, rise, up, on pay attention to "someday"
