Passage ({passage_lang_name}):
"""
{passage}
"""

Write search queries that a user could type and that this passage answers, in Kazakh, Russian and English:
- "question": one natural question per language (the three are translations of each other);
- "keywords": one short keyword query per language, as typed into a search box (the three are translations of each other), about a different fact of the passage than the question when the passage has several.

Rules:
1. Each query must be answerable from the passage alone and ask about its main facts, not about trivia in a side remark.
2. A question has 3-20 words, a keyword query 2-10 words; no query copies a full sentence of the passage.
3. Write Kazakh in Cyrillic with the Kazakh letters (ә ғ қ ң ө ұ ү һ і).
4. Output only one JSON object: {"question": {"kk": "...", "ru": "...", "en": "..."}, "keywords": {"kk": "...", "ru": "...", "en": "..."}}
