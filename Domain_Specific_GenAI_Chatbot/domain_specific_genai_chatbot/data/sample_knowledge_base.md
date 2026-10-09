# AI & Data Science Knowledge Base (Demo)

This sample document is included so the application works immediately after launch. Replace or extend it with documents from your own domain. The chatbot should answer only when its retrieved sources support the response.

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines an information-retrieval step with language-model generation. A user question is compared against chunks from a curated knowledge base. The most relevant passages are added to the model prompt, and the model uses that context to formulate an answer. RAG can make answers more relevant to private or frequently updated information without changing the model's weights.

A typical RAG pipeline includes document ingestion, text extraction, chunking, embedding or indexing, retrieval, prompt construction, answer generation, and source citation. Chunk sizes and overlap should be tested against the documents and questions in the target domain. Very large chunks can add noise; very small chunks can separate facts that belong together.

## Embeddings and Retrieval

Text embeddings are numerical representations of text that can be compared by similarity. Semantic retrieval uses an embedding model to compare the intent and meaning of a question with knowledge-base passages. TF-IDF is a lightweight lexical retrieval method that ranks documents based on term importance. It can be useful as a low-dependency fallback, but it may miss a relevant passage when the question uses different wording from the source.

The top-k setting controls how many passages are passed to the answer generator. A very small k can omit necessary information, while a large k can increase latency and introduce distracting context. Retrieval quality should be checked with representative questions and expected source passages.

## Prompt Engineering and Grounded Answers

A system prompt defines the assistant's role, boundaries, and answer style. A grounded assistant should distinguish supported facts from missing information, cite source passages, and avoid making up facts or citations. Retrieved documents should be treated as reference content rather than trusted instructions. This helps reduce the risk of prompt injection through a document.

For a domain-specific assistant, define the intended audience, vocabulary, scope, escalation rules, and topics the assistant must not answer. Test how it responds when information is missing, contradictory, outdated, or outside the domain.

## LLM Fine-Tuning

Fine-tuning updates a model using curated examples to encourage consistent behavior, formatting, or domain-specific task performance. High-quality training data should be accurate, representative, consistently formatted, and reviewed for sensitive information. Dataset examples commonly contain a user message and a preferred assistant response; compatible chat fine-tuning formats may include a system message as well.

Fine-tuning and RAG solve different problems. Fine-tuning can teach response patterns and task behavior, while RAG supplies relevant facts from a retrievable knowledge base. A system may use both, but neither guarantees factual accuracy. Training-data preparation must not be described as completed model fine-tuning unless a training job has actually run and its output model is configured.

## Evaluating a RAG Chatbot

Evaluate retrieval and generation separately. Retrieval metrics can include Recall@k and Mean Reciprocal Rank on a set of questions with labeled relevant passages. Answer evaluation can assess groundedness, relevance, completeness, citation correctness, and whether the assistant admits uncertainty when sources are insufficient. Keep a test set separate from training examples to reduce evaluation leakage.

Useful test cases include common questions, paraphrases, misspellings, questions requiring multiple passages, unsupported questions, conflicting documents, and malicious instructions embedded in source material. Log only what is necessary and avoid storing API keys or sensitive user content.

## Python and SQL Examples

Python is widely used for data processing, automation, machine learning, and application development. Pandas provides DataFrame operations for tabular data, while NumPy provides numerical arrays and mathematical functions. Scikit-learn includes utilities for feature processing, model training, and evaluation.

SQL is used to query relational data. WHERE filters rows, GROUP BY aggregates rows into groups, ORDER BY sorts results, and JOIN combines records from related tables. A reliable analytical workflow validates joins and null handling, checks aggregation grain, and compares output totals with a trusted reference.
