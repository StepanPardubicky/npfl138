### Assignment: tagger_ner
#### Date: Deadline: Apr 30, 22:00
#### Points: 2 points
#### Tests: tagger_ner_tests
#### Examples: tagger_ner_examples

This assignment is an extension of `tagger_we` task. Using the
[tagger_ner.py](https://github.com/ufal/npfl138/tree/master/labs/09/tagger_ner.py)
template, implement optimal decoding of named entity spans from
BIO-encoded tags. In a valid sequence, the `I-TYPE` tag must follow either
`B-TYPE` or `I-TYPE` tags.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

#### Tests Start: tagger_ner_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_ner.py --epochs=2 --max_sentences=2000`
```
Epoch=1/2 10.3s loss=1.0373 accuracy=0.8046 dev_loss=0.7787 dev_accuracy=0.8225 dev_f1_constrained=0.0067 dev_f1_greedy=0.0057
Epoch=2/2 9.3s loss=0.6481 accuracy=0.8179 dev_loss=0.6709 dev_accuracy=0.8331 dev_f1_constrained=0.0910 dev_f1_greedy=0.0834
```

2. `python3 tagger_ner.py --epochs=2 --max_sentences=2000 --label_smoothing=0.3`
```
Epoch=1/2 10.2s loss=1.8772 accuracy=0.8045 dev_loss=1.7571 dev_accuracy=0.8224 dev_f1_constrained=0.0039 dev_f1_greedy=0.0039
Epoch=2/2 9.3s loss=1.7125 accuracy=0.8181 dev_loss=1.7097 dev_accuracy=0.8308 dev_f1_constrained=0.0546 dev_f1_greedy=0.0413
```
#### Tests End:
#### Examples Start: tagger_ner_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_ner.py --epochs=5`
```
Epoch=1/5 47.6s loss=0.7167 accuracy=0.8339 dev_loss=0.5699 dev_accuracy=0.8422 dev_f1_constrained=0.1915 dev_f1_greedy=0.1468
Epoch=2/5 48.0s loss=0.3642 accuracy=0.8915 dev_loss=0.4467 dev_accuracy=0.8785 dev_f1_constrained=0.4204 dev_f1_greedy=0.3684
Epoch=3/5 48.3s loss=0.1773 accuracy=0.9476 dev_loss=0.4161 dev_accuracy=0.8847 dev_f1_constrained=0.4826 dev_f1_greedy=0.4307
Epoch=4/5 49.4s loss=0.0852 accuracy=0.9755 dev_loss=0.4318 dev_accuracy=0.8877 dev_f1_constrained=0.4878 dev_f1_greedy=0.4427
Epoch=5/5 49.0s loss=0.0490 accuracy=0.9860 dev_loss=0.4354 dev_accuracy=0.8948 dev_f1_constrained=0.5214 dev_f1_greedy=0.4909
```

- `python3 tagger_ner.py --epochs=5 --label_smoothing=0.3`
```
Epoch=1/5 48.6s loss=1.7265 accuracy=0.8357 dev_loss=1.6516 dev_accuracy=0.8541 dev_f1_constrained=0.2550 dev_f1_greedy=0.2130
Epoch=2/5 49.5s loss=1.5508 accuracy=0.9061 dev_loss=1.6000 dev_accuracy=0.8925 dev_f1_constrained=0.4974 dev_f1_greedy=0.4591
Epoch=3/5 50.4s loss=1.4624 accuracy=0.9618 dev_loss=1.5795 dev_accuracy=0.8993 dev_f1_constrained=0.5369 dev_f1_greedy=0.5035
Epoch=4/5 50.4s loss=1.4269 accuracy=0.9806 dev_loss=1.5778 dev_accuracy=0.9003 dev_f1_constrained=0.5475 dev_f1_greedy=0.5202
Epoch=5/5 50.0s loss=1.4109 accuracy=0.9890 dev_loss=1.5757 dev_accuracy=0.9026 dev_f1_constrained=0.5546 dev_f1_greedy=0.5368
```
#### Examples End:
