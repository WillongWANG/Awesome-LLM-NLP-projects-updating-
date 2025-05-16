# Evaluating LLMs on Natural Language Inference, hallucination detection, and Biases

The task details are in [Individual Project](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/Eval%20LMs%20for%20NLI%20%26%20biaes/CSIT_Individual_Project_2025_Spring.pdf). Refer to []() for methods and experimental settings and results. The code is modified from [Pattern-Exploiting Training](https://github.com/timoschick/pet/tree/master?tab=readme-ov-file##pet-training-and-evaluation) and [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs).

## How to run

### PET for MNLI

model: bert-base-uncased/roberta-base, task: mnli/mnli-mm (mm: mismatched)

```
python cli.py --do_train --lm_training --do_eval --method pet --model_name_or_path bert-base-uncased/roberta-base --model_type bert/roberta --task_name mnli/mnli-mm --no_distillation --pattern_ids [0,1,2,3] --pet_repetitions 1 --pet_max_steps 250 --sc_max_steps 250
```

Delete `--do_train` and `--lm_training` for evaluation only.

PLM-based `xlnet` and `xlm-roberta` model are not suitable for MNLI: 
`--model_type xlnet --model_name_or_path xlnet-large-cased --wrapper_type plm`

`--model_type xlm-roberta --model_name_or_path xlm-roberta-base`

Use Supervised Fine-Tuning instead of PET (e.g. BertForSequenceClassification, no Language Modeling):  

```
python cli.py --do_train --do_eval --method sequence_classifier --model_name_or_path bert-base-uncased --model_type bert --task_name mnli/mnli-mm --no_distillation
```


### NLI for hallucination detection

model: fine-tuned model `textattack/bert-base-uncased-MNLI` and `roberta-large-mnli` from huggingface

```
python 2.py
```


### Biases in Language Models

model: bert-base-uncased/roberta-large/albert-xxlarge-v2

```
python --input_file data\age.csv --lm_model bert/roberta/albert --outfile output\age_XXX.csv
```
