# Retrieval-Augmented Generation

Retrieval-Augmented Generation combines information retrieval with language generation. A retrieval component selects evidence from a corpus, and a language model uses that evidence to construct an answer. A useful production system should preserve source metadata so answers can expose where evidence came from.

Evaluation should separate retrieval quality from answer quality. Retrieval can be measured with ranking metrics such as recall, MRR, or NDCG, while generated answers require task-specific evaluation.
