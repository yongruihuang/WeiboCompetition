# ��������

΢��������ģԤ��
�������ӣ�http://data.sklccc.com/matchpage?t=context

΢���û�����΢���ı����Լ�΢����1Сʱ����������Ԥ��΢����24Сʱ��Ĵ�����


# ���
|  Name   | Score  | Rank  | Team member  |
|  ----  | ----  |  ----  |  ----  |
| ��������С�ֶ�  | 70.5129 | 4 |https://github.com/Chenchh12<br>https://github.com/Jaggie-Yan<br>https://github.com/yongruihuang |

# ����˵��

- 0_���ݲ鿴+���������챣���м��ļ�.ipynb:���ݷ���+�ں��û������ļ�
- 1_��������.ipynb:�����������
- 2_24h��������.ipynb��������֤����24Сʱ��������
- 3_model_lgb.ipynb����ȡ��������lgb����Ԥ��
- 4_model_nn_run.ipynb��ʹ�����������Ԥ��
- 5_model_stacking.ipynb������stacking����ģ���ں�

# ��������

## ΢���û���������

���Ƿ����û��������ͬһ�û���Ӧ������¼�����������ڲ�ͬʱ��ɼ����£����ǣ����Ƕ�ͬһ�û���ͬ��Ŀ�������ͳ�ƾۺϣ�������Ϊ�û���������

## �ı�����
�ı����������û�����΢���ı������Ƕ����߶��ֱ��������²�ͬ���ı���ȡ����
- Word2vec:ʹ��word2vec�õ�ÿ���ʵĴ����������������д�ȡ��ֵ�õ����ӵ��������������������е��ı�
- Glove:ʹ��Glove�õ�ÿ���ʵĴ����������������д�ȡ��ֵ�õ����ӵ��������������������е��ı�
- tfidf-svd������ϡ�����ķ�����ȡÿ��������tfidf������������svd��ά�õ��ı�����
- LDA���õ�ÿ�����ӵ�����ģ��������Ϊ����
- Ӳ���룺ͨ�����ݷ����ֶβ鿴΢���ı��Ƿ����ĳЩ�ؼ����硰��Ƶ�������顱��������one-hot����
- Bert��������ģ������bert�����о�����������

## ʱ������
����΢��ת����ʱ�䣬���ǽ��������ꡢ�¡��ա�Сʱ������

## һСʱת������
һСʱת����������ѵ�����Ͳ��Լ����У���˿���ֱ�����ã���Ҫ����˭ֱ��ת����Ŀ��΢����ת�����ı�
- ��Ŀ��΢���ĽǶȳ�����Groupby weibo id��ÿ�������ڵ�ת���û��Ļ�����������ͳ�ƾۺϣ�ÿ��������15mins,30mins,45mins,60mins��ת����
- ���û��Ƕȳ�����Groupby user id��ͳ�����ϵ�����

## ��������
�û�id��ʱ��Ľ��棬һСʱת��������ʱ�佻��


## 24Сʱת������
����24Сʱת�������ڲ��Լ��в����ڣ����ǣ�ֻ�ܲ�������target encoding�ķ�������ѵ��������ѵ�����Ͳ��Լ����������ǹ���ÿ��Сʱ��ת������Ϊ��������ѵ�����ֳ�5�ݣ�����ÿһ��ѵ��������<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>�����ó�ȥ<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>�������ķ����ݽ���ģ��ѵ������ÿ��Сʱת������Ϊ��ǩ������������Ϊ������ѵ�����<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>����Ԥ�⣬<img src="svgs/1a3452293037722957e353d66493e294.svg?invert_in_darkmode" align=middle width=33.36695504999999pt height=22.465723500000017pt/>���ֵõ�24Сʱ���������Բ��Լ�����Ԥ�⣬ѵ������Ԥ��ֵȡ5��Ԥ���ƽ����
���ǳ�����������ṹ����Ԥ��
- ȫ�������24\*5����Ԫ��ÿ�����Ԫ����һ��Сʱ��ת������ÿ�����Ԫ�ֱ�ȥ����������ʧ
- ����ʱ��ģ��GRU����ѵ��ʱ������**��ʵ����һ��ʱ�䲽**��Ϊ���룬����������Ϊ��ʼ��hidden status
- ����ʱ��ģ��GRU����ѵ��ʱ������**Ԥ�����һ��ʱ�䲽���**��Ϊ���룬����������Ϊ��ʼ��hidden status
- ����ʱ��ģ��GRU��Ԥ����һСʱת�����Ĳ�ֵ
- ����ʱ��ģ��GRU��Ԥ������һ��ʱ�䲽�Ĳ�ֵ


# ģ��

## lgb

������������������ȥ�ı���bert����������lgb��ģ


## ������

### ����ṹ

![avatar](images/������ṹ.png)

### ��ʧ��������

����Ŀ�꽫ת����Ϊ�����

|��λ	|ת����|Ȩ��|
|  ----  | ----  |  ----  |
|1|	0-10|	1|
|2|	11-50|	10|
|3|	51-150|	50|
|4|	151-300|	100|
|5|	300+|	300|

- ��Ȩ�����أ��Ե�λ���з��࣬<img src="svgs/82f9185dc9755b126480df473ed521ef.svg?invert_in_darkmode" align=middle width=135.48937214999998pt height=31.36100879999999pt/>��<img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>��������<img src="svgs/0dc78fafcf888221e7b04b797ecf1a81.svg?invert_in_darkmode" align=middle width=18.81483944999999pt height=14.15524440000002pt/>�����i��������j�����ʵֵ��<img src="svgs/200cdf959030dfee4638a263f63db3ae.svg?invert_in_darkmode" align=middle width=19.025975099999986pt height=14.15524440000002pt/>�����i��������j���Ԥ��ֵ��<img src="svgs/9bb9bef4064b1dd3da0feacbbbcddc96.svg?invert_in_darkmode" align=middle width=79.91057249999999pt height=24.65753399999998pt/>��<img src="svgs/40cca55dbe7b8452cf1ede03d21fe3ed.svg?invert_in_darkmode" align=middle width=17.87301779999999pt height=14.15524440000002pt/>Ϊ�����Ȩ�أ�<img src="svgs/fe54c91f95be39b9b62cae7ee59857d2.svg?invert_in_darkmode" align=middle width=174.85930439999998pt height=24.65753399999998pt/>

- �������(�ع�): ��ת�������лع飬<img src="svgs/34eb1d9a0bb904c5215114214f703118.svg?invert_in_darkmode" align=middle width=121.36256055pt height=27.91243950000002pt/>������<img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>Ϊ��ʵת�������Ļ����ֵ��<img src="svgs/5eabf65c96fa582c9314fc597a0d957b.svg?invert_in_darkmode" align=middle width=28.44140804999999pt height=27.91243950000002pt/>ΪԤ��ת����

- ������ʧ�����ڼ���������ϲ��ɷֲ�������<img src="svgs/440375cbb193942b6a1d70cd2d55fc80.svg?invert_in_darkmode" align=middle width=157.51908974999998pt height=27.91243950000002pt/>�����У�<img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>Ϊ��ʵת������<img src="svgs/5eabf65c96fa582c9314fc597a0d957b.svg?invert_in_darkmode" align=middle width=28.44140804999999pt height=27.91243950000002pt/>ΪԤ��ת����

- ��������֪ʶ�Ľ����أ���������֪��һСʱ��ת��������ˣ�����һСʱ��ת������λΪ3����ô24Сʱת����ֻ������3��4��5����˿�������mask softmax�Ļ��ƣ������������ǰ������ԪMask������С����������softmax��ǰ������Ԫ���Ϊ0������ǰ�������ĸ���Ϊ0.

- ��ԾԤ��������:��ֱ��Ԥ���ĸ���𣬶�����һСʱת���Ļ����ϣ�Ԥ����������Ծ���ٸ���λ��

### ���ǩ����

���ǵ����÷���ķ�������ת������������Ԥ�⣬��ʹ�ô��ڱ߽�ĵ�Ԥ��Ч�����ã�����300Ӧ�����ڵ��ĵ�����ʵ���еĸ��ʷֲ������ڵ��嵵�ĸ���Ҳͦ������漰��һ�ָ���ת�����õ���ǩ���ʷֲ��Ļ��ơ�ͨ���м�ĵ�λ���м��ľ��룬��ʼ�ĵ�λ�����ߵľ�����������ƣ���������


```
def get_distibution(cnt):
    '''
    Args:
        cnt:ת����
    Returns:
        ������������
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

ת������������ĸ��ʵĹ�ϵ����ͼ��ʾ

![avatar](images/���ǩ����.png)


��ˣ��޸ĺ�Ľ����ض���һ������Ϊ��

<img src="svgs/ea1ae3fa74957f26e7c6cbc5f9cd926e.svg?invert_in_darkmode" align=middle width=111.51310664999998pt height=32.256008400000006pt/>

����
- <img src="svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode" align=middle width=12.210846449999991pt height=14.15524440000002pt/>Ϊ����Ȩ�أ���ʵ��ǩ����0Ϊ1������1Ϊ10��...������4Ϊ300
- <img src="svgs/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode" align=middle width=12.92146679999999pt height=14.15524440000002pt/>Ϊģ��������߼�ֵ�����ֵsoftmax
- <img src="svgs/9294da67e8fbc8ee3f1ac635fc79c893.svg?invert_in_darkmode" align=middle width=11.989211849999991pt height=14.15524440000002pt/>Ϊ���Ƿֲ����ʣ�ÿһ��ͨ��ת������ÿһ�����ĵ���ʵ��
