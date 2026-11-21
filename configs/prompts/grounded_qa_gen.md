You prepare training data for a question answering assistant that must answer only from evidence.

Evidence:
{evidence}

Task ({mode}), write three different question-answer pairs:
- answerable: each question is a natural question in {question_lang_name} that a user could ask and that is answered by the evidence: by one passage, or by several passages only when they are about the same subject; never join unrelated facts into one question. The three questions ask about different facts. Each answer is in {question_lang_name} and cites the supporting passage after every factual sentence by its number in brackets, e.g. [2], citing only numbers listed above.
- unanswerable: each question is in {question_lang_name}, on the topic of the evidence, and asks for a plausible fact that the evidence does NOT state; the three questions ask for different missing facts. Each answer is exactly the refusal phrase: "{refusal}".

Use only facts stated in the evidence; every factual sentence of an answer must end with its citation. The evidence may be in another language than the question: translate facts faithfully and add no facts that are not in the evidence. Output only one JSON object: {"pairs": [{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}]}
