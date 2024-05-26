# Hospital Chatbot
<img src="llm.avif">

## Understand the Problem and Requirements
We want answers to ad-hoc questions about patients, visits, doctors, hospitals, and insurance payers without having to understand SQL-like query languages, request reports from analysts, or wait for someone to build dashboards.

To accomplish this, stakeholders wanted an internal chatbot tool, similar to ChatGPT, that could answer questions about company data. After meeting to gather requirements, we came up with a list of the types of questions the chatbot should answer:

- What is the current wait time at XYZ hospital?
- Which hospital currently has the shortest wait time?
- At which hospitals are patients complaining about billing and insurance issues?
- Have any patients complained about the hospital being unclean?
- What have patients said about how doctors and nurses communicate with them?
- What are patients saying about the nursing staff at XYZ hospital?
- What was the total billing amount charged to Cigna payers in 2023?
- How many patients has Dr. John Doe treated?
- How many visits are open and what is their average duration in days?
- Which physician has the lowest average visit duration in days?
- How much was billed for patient 789’s stay?
- Which hospital worked with the most Cigna patients in 2023?
- What’s the average billing amount for emergency visits by hospital?
- Which state had the largest percent increase inedicaid visits from 2022 to 2023?

## Tech Stack
- Python 3.10.11
- Langchain
- Cohere LLM
- Neo4j Database
- FastAPI (Backend)
- Streamlit (Frontend)

## Resouces
- [Neo4j Docs](https://neo4j.com/docs/getting-started/cypher-intro/cypher-sql/)
- [5 pragmatic strategies for cost optimization with llm](https://www.metadocs.co/2024/04/03/5-pragmatic-strategies-for-cost-optimization-with-llm/)