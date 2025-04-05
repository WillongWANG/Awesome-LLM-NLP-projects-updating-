# Enhancing Local Knowledge Base QA with LLMs and RAG Integration
<font size="10">
This project aims to enhance local knowledge base QA by integrating **Chinese LLMs** like ChatGLM with **RAG** technology using **LangChain**. It enables users to upload unstructured documents and generates accurate, context-aware responses through similarity search. The project plans to evaluate various prompt strategies on the JD QA dataset and some literary texts using precision and ROUGE metrics for performance assessment.

The project follows the fundamental RAG pipeline, which involves several key steps. First, the uploaded documents are split into smaller chunks based on predefined rules, making them easier to process. Next, these chunks are embedded and vectorized to facilitate similarity comparison. When a user submits a query, the system performs a similarity search on the embedded document vectors to identify the most relevant passages. The retrieved content is then combined with the user’s query to generate a coherent and accurate response.
</font>
