# Enhancing Local Knowledge Base QA with LLMs and RAG Integration

This project aims to enhance local knowledge base QA by integrating **Chinese LLMs** like ChatGLM with **RAG** technology using **LangChain**. It enables users to upload unstructured documents and generates accurate, context-aware responses through similarity search. The project plans to evaluate various prompt strategies on the JD QA dataset and some literary texts using precision and ROUGE metrics for performance assessment.

The project follows the fundamental RAG pipeline, which involves several key steps. First, the uploaded documents are split into smaller chunks based on predefined rules, making them easier to process. Next, these chunks are embedded and vectorized to facilitate similarity comparison. When a user submits a query, the system performs a similarity search on the embedded document vectors to identify the most relevant passages. The retrieved content is then combined with the user’s query to generate a coherent and accurate response.

## Chinese LLMs

1. CHATGLM

Deploy ChatGLM-6B according to the [official website](https://github.com/THUDM/ChatGLM-6B). I deployed it locally on Windows using the API and directly loaded the quantized model. The INT4 quantized model only requires about 5.2GB of memory. Model quantization may lead to some performance loss, but through testing, ChatGLM-6B can still generate natural and fluent responses with 4-bit quantization. You are recommended to download the model directly from Hugging Face ([chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/main) or [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4/tree/main)) and remember to install the GCC for the INT4 model.

```
# Replace the corresponding part in the original official api.py, where self-chatglm-6b-int4 is your local chatglm-6b or chatglm-6b-int4 directory
tokenizer = AutoTokenizer.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True).half().cuda()
model.eval()  
```

Then, deploy the ChatGLM model:

```
# first create a virtual environment for deploying
pip install -r requirements_chatglm.txt
python api.py
```

Create the LLM using the following approach in chatglm_document_qa.py:

```
from langchain_community.llms import ChatGLM

llm = ChatGLM(
     endpoint_url='http://127.0.0.1:8000',
     max_token=80000,
     top_p=0.9)
```

2. DeepSeek

To balance time and cost, I also used the DeepSeek API. According to the official website, the deepseek-chat model has been fully upgraded to DeepSeek-V3.

```
from langchain_deepseek import ChatDeepSeek

llm=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2000,
    api_key="") #Your api_key
```

## How to run

```
pip install -r requirements.txt
python chatglm_document_qa.py
```

Default settings:

You can adjust them accordingly.

```
text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0) #chunk_size & chunk_overlap to split the original or uploaded text

# prompt
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据以下已知信息回答问题：
{context}
问题：{question}
请用中文简洁回答，如果无法回答请说不知道。""")

retriever = db.as_retriever(search_kwargs={"k": 3}) #return 3 relevant docs
```






