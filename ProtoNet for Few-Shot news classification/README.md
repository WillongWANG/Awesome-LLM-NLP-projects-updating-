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
```
Selected classes: ['游戏', '股票', '政治', '财经']
Query sentences: ['《超级马里奥3DS》揭晓 玩法缤纷多彩', 'PSN益智游戏《Okabu》首支宣传片公布', '《绿日：摇滚乐队》新游戏宣传片公布', '《风华之歌》精彩介绍资料抢先放送', '《最终幻想 零式》召唤兽详情公布', '雷普索尔将以6.39亿美元出售YPF公司3.83%股份', '7间大行对利丰最新投资建议一览', '趋势交易高手 时点仓位执行缺一不可', '港股午后中资钢铁股跑赢大市', '旺季刺激消费升级 纺织编织上涨空间', '也门政府谴责胡塞武装违反停火协议', '习近平在中俄执政党对话机制会议开幕式讲话', '巴基斯坦部落地区遭无人机袭击4人死亡', '全球首辆飞天汽车下月试飞 时速超100公里(图)', '韩国媒体称天安舰系朝鲜事先计划击沉', '基金去年托管费超50亿 工行建行占据半壁江山', '商品熊市会否因收储而改变', '证监会再批三只新基金', '中海货币基金首发募集近30亿', '宝盈基金习惯性垫底 一月业绩排名倒数第一']
Loss: 0.4865 Acc: 0.8000
```
