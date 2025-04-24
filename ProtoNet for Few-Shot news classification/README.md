This project uses [Prototypical Networks](https://arxiv.org/pdf/1703.05175) for few-shot news headline classification, where the classes are new and unseen in the training set.

![](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/ProtoNet%20for%20Few-Shot%20news%20classification/1.png)

The training dataset contains a title and its corresponding class in each line, with data stored in [train.txt](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/ProtoNet%20for%20Few-Shot%20news%20classification/data/train.txt), consisting of 10 categories from [class.txt](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/blob/main/ProtoNet%20for%20Few-Shot%20news%20classification/data/thunews/class.txt). test.txt only have 4 categories.

The default setup is 4-way 5-shot in both train.py and test.py:  
--num_way=4  
--num_shots=5  

In test.py:  
--num_queries=5,  
--seed=10,  #7, 10 may have better results  

Run:  
```
python train.py
```  
```
python test.py
```

Example results:  
