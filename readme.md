# 比赛背景

微博传播规模预测
比赛链接：http://data.sklccc.com/matchpage?t=context

微博用户画像、微博文本，以及微博的1小时传播特征，预测微博在24小时后的传播量


# 结果
|  Name   | Score  | Rank  | Team member  |
|  ----  | ----  |  ----  |  ----  |
| 长城拿铁小分队  | 70.5129 | 4 |https://github.com/Chenchh12<br>https://github.com/Jaggie-Yan<br>https://github.com/yongruihuang |

# 代码说明

- 0_数据查看+简单特征构造保存中间文件.ipynb:数据分析+融合用户画像文件
- 1_特征构造.ipynb:特征构造代码
- 2_24h特征构造.ipynb：交叉验证利用24小时特征代码
- 3_model_lgb.ipynb：抽取特征利用lgb进行预测
- 4_model_nn_run.ipynb：使用神经网络进行预测
- 5_model_stacking.ipynb：利用stacking进行模型融合

# 特征构造

## 微博用户画像特征

我们发现用户画像表中同一用户对应多条记录，可能是由于不同时间采集所致，于是，我们对同一用户不同条目将其进行统计聚合，用来作为用户画像特征

## 文本特征
文本特征包括用户简介和微博文本，我们对两者都分别尝试了以下不同的文本抽取方法
- Word2vec:使用word2vec得到每个词的词向量，将句子所有词取均值得到句子的向量，用来表征样本中的文本
- Glove:使用Glove得到每个词的词向量，将句子所有词取均值得到句子的向量，用来表征样本中的文本
- tfidf-svd：利用稀疏矩阵的方法抽取每个样本的tfidf向量，再利用svd降维得到文本特征
- 硬编码：通过数据分析手段查看微博文本是否出现某些关键词如“视频”“疫情”，将其变成one-hot向量
- Bert：神经网络模型中用bert来进行句子向量表征

## 时间特征
包括微博转发的时间，我们将其编码成年、月、日、小时、周天

## 一小时转发特征
一小时转发特征由于训练集和测试集都有，因此可以直接利用，主要包括谁直接转发了目标微博和转发的文本
- 从目标微博的角度出发：Groupby weibo id，每个分组内的转发用户的画像特征进行统计聚合，每个分组内15mins,30mins,45mins,60mins的转发量
- 从用户角度出发，Groupby user id，统计如上的特征

## 交叉特征
用户id和时间的交叉，一小时转发数量和时间交叉


## 24小时转发特征
由于24小时转发数据在测试集中不存在，于是，只能采用类似target encoding的方法交叉训练集构造训练集和测试集特征，我们构造每个小时的转发量作为特征，将训练集分成5份，遍历每一份训练集数据<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>，利用除去<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>的其他四份数据进行模型训练，用每个小时转发数作为标签，其他特征作为特征，训练后对<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>进行预测，<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>部分得到24小时的特征，对测试集进行预测，测试集的预测值取5次预测的平均。
我们尝试以下网络结构进行预测
- 全连接输出23\*5个神经元，每五个神经元代表一个小时的转发量，每五个神经元分别去做交叉熵损失
- 利用时序模型GRU，在训练时候利用**真实的上一步时间步**作为输入，其他特征作为初始的hidden status
- 利用时序模型GRU，在训练时候利用**预测的上一步时间步输出**作为输入，其他特征作为初始的hidden status
- 利用时序模型GRU，预测与一小时转发量的差值
- 利用时序模型GRU，预测与上一个时间步的差值


# 模型

## lgb

将上述所有特征（除去文本的bert特征）利用lgb建模


## 神经网络

### 网络结构

![avatar](images/神经网络结构.png)

### 损失函数构造

评价目标将转发量为五分类

|档位	|转发数|权重|
|  ----  | ----  |  ----  |
|1|	0-10|	1|
|2|	11-50|	10|
|3|	51-150|	50|
|4|	151-300|	100|
|5|	300+|	300|

- 带权交叉熵：对档位进行分类，<img src="svgs/82f9185dc9755b126480df473ed521ef.svg?invert_in_darkmode" align=middle width=135.48937214999998pt height=31.36100879999999pt/>，<img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>个样本，<img src="svgs/0dc78fafcf888221e7b04b797ecf1a81.svg?invert_in_darkmode" align=middle width=18.81483944999999pt height=14.15524440000002pt/>代表第i个样本第j类的真实值，<img src="svgs/200cdf959030dfee4638a263f63db3ae.svg?invert_in_darkmode" align=middle width=19.025975099999986pt height=14.15524440000002pt/>代表第i个样本第j类的预测值。<img src="svgs/9bb9bef4064b1dd3da0feacbbbcddc96.svg?invert_in_darkmode" align=middle width=79.91057249999999pt height=24.65753399999998pt/>，<img src="svgs/40cca55dbe7b8452cf1ede03d21fe3ed.svg?invert_in_darkmode" align=middle width=17.87301779999999pt height=14.15524440000002pt/>为该类的权重，<img src="svgs/fe54c91f95be39b9b62cae7ee59857d2.svg?invert_in_darkmode" align=middle width=174.85930439999998pt height=24.65753399999998pt/>

- 均方误差(回归): 对转发量进行回归，<img src="svgs/34eb1d9a0bb904c5215114214f703118.svg?invert_in_darkmode" align=middle width=121.36256055pt height=27.91243950000002pt/>，其中<img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>为真实转发量中心化后的值，<img src="svgs/5eabf65c96fa582c9314fc597a0d957b.svg?invert_in_darkmode" align=middle width=28.44140804999999pt height=27.91243950000002pt/>为预测转发量

- 泊松损失：由于计数问题符合泊松分布，采用<img src="svgs/440375cbb193942b6a1d70cd2d55fc80.svg?invert_in_darkmode" align=middle width=157.51908974999998pt height=27.91243950000002pt/>，其中，<img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>为真实转发量，<img src="svgs/5eabf65c96fa582c9314fc597a0d957b.svg?invert_in_darkmode" align=middle width=28.44140804999999pt height=27.91243950000002pt/>为预测转发量

- 引入先验知识的交叉熵：由于我们知道一小时的转发量，因此，比如一小时的转发量挡位为3，那么24小时转发量只可能是3、4、5，因此可以引入mask softmax的机制，将神经网络输出前两个神经元Mask成无穷小，这样经过softmax后，前两个神经元输出为0，代表前两个类别的概率为0.

- 跳跃预估交叉熵:不直接预估哪个类别，而是在一小时转发的基础上，预估会向上跳跃多少个档位。

### 软标签构造

考虑到利用分类的方法来对转发量分箱后进行预测，会使得处于边界的点预测效果不好，比如300应该属于第四档，但实际中的概率分布，属于第五档的概率也挺大，因此设计了一种根据转发量得到标签概率分布的机制。通过使得中间的档位的距离为：转发量与中间点的距离，起始的档位的距离为：转发量与两边的距离来进行设计，代码如下

```
def get_distibution(cnt):
    '''
    Args:
        cnt:转发数
    Returns:
        五个类概率向量
    '''
    ret = np.zeros(5)
    if cnt > 400:
        cnt = 400
    ll = [-10,11,51,151,301]
    rr = [10,50,150,300,500]
    median = [(l+r)//2 for l,r in zip(ll, rr)]
    interval_length = [r-l+3 for l, r in zip(ll, rr)]
    eps = 1e-8
    for i in range(5):
        dis = abs(cnt - median[i])/interval_length[i]
        ret[i] = 1/(dis+eps)
    return ret/sum(ret)
```

转发数和五个类别的概率的关系如下图所示

![avatar](images/软标签性质.png)


因此，修改后的交叉熵对于一个样本为：

<img src="svgs/ea1ae3fa74957f26e7c6cbc5f9cd926e.svg?invert_in_darkmode" align=middle width=111.51310664999998pt height=32.256008400000006pt/>

其中
- <img src="svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode" align=middle width=12.210846449999991pt height=14.15524440000002pt/>为样本权重，真实标签属于0为1，属于1为10，...，属于4为300
- <img src="svgs/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode" align=middle width=12.92146679999999pt height=14.15524440000002pt/>为模型输出的逻辑值，输出值softmax
- <img src="svgs/9294da67e8fbc8ee3f1ac635fc79c893.svg?invert_in_darkmode" align=middle width=11.989211849999991pt height=14.15524440000002pt/>为真实分布概率，每一类通过转发数和每一类距离的倒数实现
