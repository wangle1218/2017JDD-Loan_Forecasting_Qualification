
# JDD-2017京东金融全球数据探索者大赛，信贷需求预测赛题总结

经过12月上半旬的半个月的激战，这次比赛的成绩并不理想，但是作为第一次参与的这样的正式的比赛也算学习到比较多的知识了，时间没有白费。对于这次比赛，会在***接下来的几天*** \(**已经录制完**)将所写的代码进行重构，顺便参考一位排名17选手的开源代码和提特征思路；同时也会录制一系列完整的视频，记录这个过程，我想这有助于不会写竞赛baseline代码的同学参考。

[视频地址](https://www.bilibili.com/video/av17780394/)

本次对代码进行重构的过程中也学习同不少知识，同时也对之前的竞赛提取特征思路进行了一些修正，使线下成绩有了不少提升；比如仅使用 t\_user.csv 和 t\_loan.csv两个文件就能使线下成绩达到 1.7929，在比赛的时候就听说可以仅使用这两张表就可以达到1.80，1.79，当时觉得挺难，没想到将自己的代码改一下也可以达到；另外再加上 t\_order.csv和 t\_click.csv这两个表的单表特征（没有提取交叉特征）成绩就可以提升到1.7877（还没有经过调参和模型融合），这个成绩已经比我比赛时的最终线下成绩1.789（线上成绩B榜1.7744，A榜1.7970）要好了。

关于这次比赛的一些思考，前几天在我的知乎专栏也写了一篇文章，看看可能会对理解竞赛和这个赛题有所帮助，[文章地址](https://zhuanlan.zhihu.com/p/32354021)


## 重构代码中的函数以及特征变量命名解释：

### gen\_train\_feat.py : 提取训练集特征的脚本（8，9，10月的数据）

### gen\_test\_feat.py : 提取测试集特横的脚本（9，10，11月的数据）

### util.py : 用到的一些辅助工具函数

### train.py : 训练脚本

---
<table>

<tr>
<td>**函数/特征名**</td>
<td>**解释**</td>
<td>**备注**</td>
</tr>


<tr>
<td><font color=red size=5 face=“黑体”>gen\_user\_feat()</font></td>
<td>使用 t\_user.csv表提取用户固有属性</td>
<td></td>
</tr>


<tr>
<td>a\_date</td>
<td>将激活日期转换成激活日距离预测月份的时间长度</td>
<td>训练集和测试集分别为距离11月1日和12月1日的时间长度，时间粒度为**周**</td>
</tr>


<tr>
<td>limit</td>
<td>用户的初始贷款额度，转换成实际金额</td>
<td></td>
</tr>


<tr>
<td><font color=red size=5 face=“黑体”>gen\_loan\_feat()</font></td>
<td>使用 t_loan.csv 文件提取历史贷款行为的基本业务逻辑特征</td>
<td>训练集中去掉11月的数据，测试集中去掉8月的数据</td>
</tr>


<tr>
<td>month</td>
<td>从贷款时间变量中提取的贷款月份</td>
<td></td>
</tr>


<tr>
<td>loan\_time\_hours</td>
<td>从贷款时间变量中提取的贷款时间（小时）特征</td>
<td>按照该属性提取用户贷款在一天24小时中的贷款时间喜好分布特征，将24小时平均划分成6个时间段</td>
</tr>


<tr>
<td>loan\_hours\_0x</td>
<td>0x时间段，四个小时</td>
<td>提取0x时间段内某用户贷款的次数占其所有贷款次数的占比（非贷款次数）</td>
</tr>


<tr>
<td>loan\_sum\_hour</td>
<td>用户在所有时间段的贷款次数之和</td>
<td>作为辅助变量来计算每个时间段的占比，最后删除</td>
</tr>

<tr>
<td>statistic\_df</td>
<td>用户在三个月中每个月的统计特征，'min','mean','max','std','median'</td>
<td></td>
</tr>

<tr>
<td>plannum\_0x</td>
<td>用户贷款分期的喜好特征</td>
<td>计算用户在1，3，6，12四种分期行为中偏好哪一种分期，计算占比</td>
</tr>


<tr>
<td>last\_loan\_time</td>
<td>用户每次贷款的贷款时间距离预测月份的周数</td>
<td>保留最后一次的贷款行为，删除其他时间的贷款记录就得到了每一个用户的最后一次贷款时间距离预测月份的周数</td>
</tr>


<tr>
<td>per\_loan\_time\_interval</td>
<td>用户平均贷款时间间隔</td>
<td>计算用户平均隔多久会贷款，如果仅有一次贷款行为，那么时间间隔就为贷款时间和8月1日的距离；如果多余1次贷款行为，那么间隔就为多次贷款之间间隔的平均值。时间单位为周</td>
</tr>


<tr>
<td>is\_exceed\_loan\_interval</td>
<td>最后一次贷款离预测月份的时间间隔是否超过平均贷款时间间隔</td>
<td>单位为周</td>
</tr>


<tr>
<td>exceed\_loan\_x</td>
<td>计算每个用户在每个月的每个单次贷款的贷款金额是否超过他的限额limit</td>
<td>用户初始限额为limit</td>
</tr>

<tr>
<td>new\_limit</td>
<td>新的用户贷款限额</td>
<td>根据exceed\_loan\_x，如果单次贷款额度多余初始限额，那么就说明该用户的限额提高了，而且提高后的额度至少是该单次的贷款额度</td>
</tr>

<tr>
<td>active\_loan\_interval</td>
<td>用户首次贷款的时间和该用户的激活时间的间隔</td>
<td>时间单位为周</td>
</tr>

<tr>
<td>loanTime\_weights</td>
<td>贷款时间权重</td>
<td>由于贷款时间不同，因为对于不同时间的贷款，这个贷款的影响对用户在预测月份的影响也会不同，直觉觉得影响权重会随着时间的延长而衰减</td>
</tr>

<tr>
<td>loan\_weights</td>
<td>贷款权重</td>
<td>贷款金额乘以贷款时间的权重</td>
</tr>

<tr>
<td>loan\_times\_months</td>
<td>三个月内用户在其中几个月有贷款行为</td>
<td>取值范围是{0，1，2，3}</td>
</tr>

<tr>
<td>loan\_xy</td>
<td>第x月和第y月是否都贷款了</td>
<td>都贷款了取值为1，反之为0</td>
</tr>

<tr>
<td>per\_plannum\_loan</td>
<td>平均每个分期内的贷款金额</td>
<td>贷款总金额除以分期的总期数</td>
</tr>

<tr>
<td>per\_times\_loan</td>
<td>有贷款的月份的平均贷款金额</td>
<td>贷款总金额除以贷款月份的月数</td>
</tr>

<tr>
<td><font color=red size=5 face=“黑体”>gen\_filter\_loan\_feat()</font></td>
<td>提取时间序列内的固定时间窗口内的统计特征 ['min','mean','max','median','count','sum','std']</td>
<td>[0,3,7,14,21,28,35,42,49,56,63,70,77,84]</td>
</tr>

<tr>
<td><font color=red size=5 face=“黑体”>gen\_click\_feat()</td>
<td>使用 t\_click.csv文件提取点击行为特征</td>
<td></td>
</tr>

<tr>
<td>click\_time\_hours</td>
<td>利用点击时间特征提取点击时间（小时）</td>
<td>和t\_loan.csv文件的处理一样，提取点击时间分布的占比特征</td>
</tr>

<tr>
<td>....</td>
<td>....</td>
<td>每个文件提取的相类似的特征，解释省略，具体看代码</td>
</tr>

<tr>
<td>click\_weights</td>
<td>点击权重</td>
<td>时序变量中，点击行为对预测月份的影响随着时间的延长而衰减</td>
</tr>

<tr>
<td><font color=red size=5 face=“黑体”>gen\_order\_feat()</font></td>
<td>使用 t\_order.csv文件提取用户的消费属性特征</td>
<td></td>
</tr>

<tr>
<td>real\_price</td>
<td>使用购买商品的实际消费金额</td>
<td>商品单价乘以数量减去折扣掉的金额</td>
</tr>

<tr>
<td>....</td>
<td>....</td>
<td>每个文件提取的相类似的特征，解释省略，具体看代码</td>
</tr>

<tr>
<td>buy\_weights</td>
<td>购买商品的消费金额权重</td>
<td>随着时间的延长，消费金额对用户的影响将衰弱</td>
</tr>

<tr>
<td><font color=red size=5 face=“黑体”>gen\_labels()</font></td>
<td>使用t\_loan_\sum.csv文件作为训练集的标签label</td>
<td>训练集用来预测11月的贷款金额，使用11月的汇总金额作为标签</td>
</tr>

<tr>
<td>make\_training\_data()</font></td>
<td>构造训练数据及，将前面的几张表所提取的特征进行拼接成一张表，作为模型的输入</td>
<td></td>
</tr>

<tr>
<td><font color=red size=5 face=“黑体”>make\_test\_data()</font></td>
<td>构造测试数据集</td>
<td></td>
</tr>

<tr>
<td></td>
<td></td>
<td></td>
</tr>

