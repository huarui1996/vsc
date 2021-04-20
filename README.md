# 融合语义信息的视频摘要生成
文章demo已上传至github，您可以下载观看。  
接下来将介绍项目中其他文件：  

##    关于数据
###   视频数据  
视频数据来源为通用的GoogLeNet提取好的特征，链接在[vsumm-reinforce](https://github.com/huarui1996/pytorch-vsumm-reinforce)。  
之前各个项目的issue里都有很多人表示找不到SumMe和TvSum的原始数据，  
这里我分别将它们找到了并上传到了百度网盘：  
[SumMe](https://pan.baidu.com/s/1OzC788NS5RV5YeKf5KLZAQ) 提取码: ap48 [TVSum](https://pan.baidu.com/s/1iKh2jxr71f82w87rdg5UzA) 提取码: 3ywy  
希望对大家有帮助！  
###   文本数据  
经我们标注caption后的文本数据已上传，命名为text-annotation。  
  
##    关于代码
VideoSumandCaption文件夹中存放着本文代码：  
###   模型代码  
模型代码存放在models文件夹中，是按模块编写的，清晰易懂。
###   主要代码
* run_new.py中包含3个mode，pretrain、train、test，  
运行时按如下规则：  
```bash
python run_new.py --mode train --learning_rate 1e-4 --batch_size 32 --epochs 30
```
* vsc_exp_0.yaml文件中可以修改特征维度、loss权重等参数，无需在代码文件中逐一进行修改。  
* tools.py中存放的是摘要生成以及评估代码，来源于[vsumm-reinforce](https://github.com/huarui1996/pytorch-vsumm-reinforce)。
* knapsack.py中存放的是01背包算法代码。  
###   其他代码  
文件夹中的其他代码是对数据以及msrvtt各种处理时用到的代码，并非文章核心部分。  

由于无法上传大文件至github，所以其实只有开源代码而没有处理后的各类数据。  
大家研究时只参考与文章相关的代码即可。
