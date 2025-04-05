# Enhancing Local Knowledge Base QA with LLMs and RAG Integration

This project aims to enhance local knowledge base QA by integrating **Chinese LLMs** like ChatGLM with **RAG** technology using **LangChain**. It enables users to upload unstructured documents and generates accurate, context-aware responses through similarity search. The project plans to evaluate various prompt strategies on the JD QA dataset and some literary texts using precision and ROUGE metrics for performance assessment.

The project follows the fundamental RAG pipeline, which involves several key steps. First, the uploaded documents are split into smaller chunks based on predefined rules, making them easier to process. Next, these chunks are embedded and vectorized to facilitate similarity comparison. When a user submits a query, the system performs a similarity search on the embedded document vectors to identify the most relevant passages. The retrieved content is then combined with the user’s query to generate a coherent and accurate response.

## Chinese LLMs

1. CHATGLM

Deploy ChatGLM-6B according to the [official website](https://github.com/THUDM/ChatGLM-6B). I deployed it locally on Windows using the API and directly loaded the quantized model. The INT4 quantized model only requires about 5.2GB of memory. Model quantization may lead to some performance loss, but through testing, ChatGLM-6B can still generate natural and fluent responses with 4-bit quantization. You are recommended to download the model directly from Hugging Face ([chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/main) or [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4/tree/main)) and remember to install GCC for the INT4 model.

```
# Replace the corresponding part in the original official api.py, where self-chatglm-6b-int4 is your local chatglm-6b or chatglm-6b-int4 directory
tokenizer = AutoTokenizer.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True).half().cuda()
model.eval()  
```

Then, deploy it by running:

```
python api.py
```

