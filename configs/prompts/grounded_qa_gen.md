You prepare training data for a question answering assistant that must answer only from evidence.

Evidence:
{evidence}

Task ({mode}):
- answerable: write one natural question in {question_lang_name} that a user could ask and that is answered by the evidence: by one passage, or by several passages only when they are about the same subject; never join unrelated facts into one question. Then write the answer in {question_lang_name}. Cite the supporting passage after every factual sentence as [chunk_id], citing only ids listed above.
- unanswerable: write a question in {question_lang_name} on the same topic that the evidence does NOT answer (a plausible but missing fact). The answer is exactly the refusal phrase: "{refusal}".

Use only facts stated in the evidence; every factual sentence of an answer must end with its citation. The evidence may be in another language than the question: translate facts faithfully and add no facts that are not in the evidence. Output only one JSON object: {"question": "...", "answer": "..."}
