### Lecture: 9. Structured Prediction, CTC, Word2Vec
#### Date: Apr 15
#### Slides: https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/slides/?09
#### Reading: https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/slides.pdf/npfl138-2324-09.pdf, PDF Slides
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl138/2324/npfl138-2324-09-czech.mp4, CZ Lecture
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl138/2324/npfl138-2324-09-czech.practicals.mp4, CZ Practicals
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl138/2324/npfl138-2324-09-english.mp4, EN Lecture
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl138/2324/npfl138-2324-09-english.practicals.mp4, EN Practicals
#### Questions: #lecture_9_questions
#### Lecture assignment: tensorboard_projector
#### Lecture assignment: tagger_ner
#### Lecture assignment: ctc_loss
#### Lecture assignment: speech_recognition

- Structured prediction
- Connectionist Temporal Classification (CTC) loss [[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)]
- `Word2vec` word embeddings, notably the CBOW and Skip-gram architectures [[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)]
  - Hierarchical softmax [Section 12.4.3.2 of DLB or [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
  - Negative sampling [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
- _Character-level embeddings using character n-grams [Described simultaneously in several papers as Charagram ([Charagram: Embedding Words and Sentences via Character n-grams](https://arxiv.org/abs/1607.02789)), Subword Information ([Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) or SubGram ([SubGram: Extending Skip-Gram Word Representation with Substrings](http://link.springer.com/chapter/10.1007/978-3-319-45510-5_21))]_
- _ELMO [[Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer: Deep contextualized word representations](https://arxiv.org/abs/1802.05365)]_
