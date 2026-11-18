Passage ({passage_lang_name}):
"""
{passage}
"""

Write three search queries that a user could type and that this passage answers: one in Kazakh, one in Russian and one in English.

Rules:
1. Each query must be answerable from the passage alone and ask about its main facts, not about trivia in a side remark.
2. A query is a natural question or keyword query of 3-20 words; it must not copy a full sentence of the passage.
3. Write Kazakh in Cyrillic with the Kazakh letters (ә ғ қ ң ө ұ ү һ і).
4. Output only one JSON object: {"kk": "...", "ru": "...", "en": "..."}
