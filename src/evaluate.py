"""Answer-quality and retrieval evaluation.

Score the system on a small test set and write the results to reports/answer_quality.md:
    - retrieval relevance: for sample profiles, are the top jobs actually relevant? (hit rate)
    - answer quality: correctness, grounding, helpfulness of mentor answers
    - hallucination check: does the mentor refuse when the answer is not in the notes?
"""
