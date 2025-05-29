# Parameter-Efficient Fine-Tuning ChatGLM with [Low-Rank Adaptation](https://arxiv.org/pdf/2106.09685) and [P-Tuning v2](https://aclanthology.org/2022.acl-short.8.pdf)

The code is modified from [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main).

## P-Tuning v2

Run [demo.ipynb](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/ChatGLM/Taobao_AdvertiseGen/demo.ipynb) to use pretrained chatglm-6b with your modifications:  
```
AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, cache_dir=".../chatglm6b-local")
AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, cache_dir=".../chatglm6b-local")
tokenizer = AutoTokenizer.from_pretrained(".../chatglm6b-local/models--THUDM--chatglm-6b/snapshots/...", trust_remote_code=True)
model = AutoModel.from_pretrained("/root/chatglm6b-local/models--THUDM--chatglm-6b/snapshots/...", trust_remote_code=True).half().cuda()
```

Run [ptuning.ipynb](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/ChatGLM/Taobao_AdvertiseGen/ptuning.ipynb) to P-tuning chatglm-6b on Taobao advertisement dataset. Results are included. Remember to replace the `model_name_or_path` in [train.sh](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/ChatGLM/ChatGLM-6B/ptuning/train.sh) and [evaluate.sh](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/ChatGLM/ChatGLM-6B/ptuning/evaluate.sh).

## LoRA

Run [train.py](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/zero_nlp/simple_thu_chatglm6b/train.py) with the same dataset and modifications mentioned above with the following requirements:  
```
protobuf==3.20.0
icetk
transformers==4.27.1
cpm_kernels peft==0.2.0
```  
I do not use the [chatglm6b-dddd](https://huggingface.co/yuanzhoulvpi/chatglm6b-dddd) in the orginal code as unexpected errors occur and adopt the original `THUDM/chatglm-6b` instead.  

Then run [infer.ipynb](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/PEFT%20ChatGLM%20with%20LoRA%20%26%20P-tuning%20v2/zero_nlp/simple_thu_chatglm6b/infer.ipynb) with the same modifications above to evaluate LoRA fine-tuned chatglm-6b.
