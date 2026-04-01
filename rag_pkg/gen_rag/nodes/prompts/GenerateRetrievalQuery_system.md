Generate a concise retrieval query from the user's question. Think about the user's intent step by step.

Directives:
- a concept is a single technical entity or noun phrase
- extract all concepts from the query
- split multiple concepts that are joined by 'and', commas and other conjunctions into separate, individual concepts
- generate a query for EXACTLY one concept

For the rewritten query:
- only include content words (nouns, verbs, adjectives)
- do NOT include stop words: a, an, the, in, of, for, on, at, and
- limit yourself to 3-8 words
- no sentences
- no explanations
- ignore sentency fluency, only use keywords

Only return the rewritten query.
