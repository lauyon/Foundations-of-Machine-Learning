## 1 Introduction
Machine learning can be broadly deﬁned as computational methods using experience to improve performance or to make accurate predictions. 

机器学习可以被广泛地定义为使用经验来提高性能或做出准确预测的计算方法。

- accurate adj. 正确无误的;精确的;准确的;准确的(掷、射、击等)
- prediction n.预言;预测;预告



Here, experience **refers  to** the past information **available to** the learner, which typically **takes the form of** electronic data collected and made available for analysis. 

在这里，经验是指学习者可以获得的过去的信息，这些信息通常以收集并提供用于分析的电子数据的形式出现。

- takes the form of 以...方式出现
- refers to ...是指...
- available to 可获得的



This data could be in the form of digitized human-labeled training sets, or other types of information obtained via interaction with the environment. 

这些数据可以是数字化的人工标记训练集的形式，也可以是通过与环境交互获得的其他类型的信息。

- via prep.经由，经过(某一地方);通过，凭借(某人、系统等)



In all cases, its quality and size are crucial to the success of the predictions made by the learner. 

在所有情况下，它的质量和大小对学习者做出的预测的成功至关重要。



Machine learning consists of **designing eﬃcient** and **accurate prediction algorithms**.

机器学习包括设计高效和准确的预测算法。



 As in other areas of computer science, some critical measures of the quality of these algorithms are their time and space **complexity**.

与计算机科学的其他领域一样，衡量这些算法质量的一些关键指标是它们的时间和空间复杂性。

- complexity 复杂度



 But, in machine learning, we will need additionally a **notion** of sample complexity to evaluate the sample size required for the algorithm to learn a family of concepts. 

但是，在机器学习中，我们还需要一个**样本复杂性**的概念来评估算法学习一系列概念所需的样本量。

- notion 概念



More generally, theoretical learning guarantees for an algorithm depend on the complexity of the concept classes considered and the size of the training sample.

一般来说，算法的理论学习保证取决于所考虑的概念类的复杂性和训练样本的大小。

- guarantee 保证
- theoretical learning guarantees for an algorithm 算法的理论学习



 Since the success of a learning algorithm depends on the data used, machine learning is inherently related to data analysis and statistics. 

由于学习算法的成功取决于使用的数据，机器学习本质上与数据分析和统计有关。

- inherently adv.内在的，天性地，固有地 



More generally, learning techniques are data-driven methods combining fundamental concepts in computer science with ideas from statistics, probability and optimization.

更一般地说，学习技术是数据驱动的方法，将计算机科学中的基本概念与统计、概率和优化的思想结合起来。

- optimization n.优化



### 1.1 Applications and problems
Learning algorithms have been successfully deployed in a variety of applications, including

学习算法已成功地部署在各种应用程序中，包括

- Text or document classiﬁcation, e.g., spam detection; 
- 文本或文档分类，例如垃圾邮件检测；
- Natural language processing, e.g., morphological analysis, part-of-speech tagging, statistical parsing, named-entity recognition; 
- 自然语言处理，如形态分析、部分语音标记、统计分析、命名实体识别；
- Speech recognition, speech synthesis, speaker veriﬁcation; 
- 语音识别、语音合成、扬声器验证；
- Optical character recognition (OCR); 
- 光学字符识别（OCR）；
- Computational biology applications, e.g., protein function or structured prediction;
- 计算生物学应用，例如蛋白质功能或结构化预测；
- Computer vision tasks, e.g., image recognition, face detection; 
- 计算机视觉任务，如图像识别、人脸检测；
- Fraud detection (credit card, telephone) and network intrusion;
- 欺诈检测（信用卡、电话）和网络入侵；
- Games, e.g., chess, backgammon;
- 游戏，如国际象棋、双陆棋；
- Unassisted vehicle control (robots, navigation);
- 无辅助车辆控制（机器人、导航）；
- Medical diagnosis; 
- 医学诊断；
- Recommendation systems, search engines, information extraction systems. 
- 推荐系统，搜索引擎，信息提取系统。

This list is by no means comprehensive, and learning algorithms are applied to new applications every day. 

这个列表决不是全面的，学习算法每天都应用于新的应用程序。

Moreover, such applications correspond to a wide variety of learning problems.

此外，这种应用程序对应于各种各样的学习问题。

 Some major classes of learning problems are: 

一些主要的学习问题是：

- *Classiﬁcation*: Assign a category to each item. For example, document classiﬁcation may assign items with categories such as politics, business, sports, or weather while image classiﬁcation may assign items with categories such as landscape, portrait, or animal. The number of categories in such tasks is often relatively small, but can be large in some diﬃcult tasks and even unbounded as in OCR, text classiﬁcation, or speech recognition. 

- 分类：为每个项目分配一个类别。例如，文档分类可以为项目分配政治、商业、体育或天气等类别，而图像分类可以为项目分配景观、肖像或图像等类别。这类任务中的类别数量通常相对较少，但在一些困难的任务中可能较大，甚至在OCR、文本分类或语音识别中也不受限制。

   + Assign  分配

- *Regression*: Predict a real value for each item. Examples of regression include prediction of stock values or variations of economic variables. In this problem, the penalty for an incorrect prediction depends on the magnitude of the diﬀerence between the true and predicted values, in contrast with the classiﬁcation problem, where there is typically no notion of closeness between various categories. 

- 回归：预测每个项目的实际值。回归的例子包括股票价值的预测或经济变量的变化。在这个问题中，错误预测的惩罚取决于真实值和预测值之间差异的大小，与分类问题不同，分类问题通常不存在不同类别之间的紧密性概念。

  + magnitude n.巨大;重大;重要性;星等;星的亮度;震级

- *Ranking*: Order items according to some criterion. Web search, e.g., returning web pages relevant to a search query, is the canonical ranking example. Many other similar ranking problems arise in the context of the design of information extraction or natural language processing systems.

- 排序：根据某种标准对项目进行排序。Web搜索（例如返回与搜索查询相关的网页）是规范的排名示例。在信息提取或自然语言处理系统的设计中，也出现了许多类似的排序问题。

  + criterion 标准
  + canonical 规范的

-  *Clustering*: Partition items into homogeneous regions. Clustering is often performed to analyze very large data sets. For example, in the context of social network analysis, clustering algorithms attempt to identify “communities” within large groups of people. 

- 聚类：将项目划分为均匀区域。聚类通常用于分析非常大的数据集。例如，在社会网络分析的背景下，聚类算法试图识别大群体中的“社区”。

  + partition 分割
  + homogeneous 同类的

- *Dimensionality reduction or manifold learning*: Transform an initial representation of items into a lower-dimensional representation of these items while preserving some properties of the initial representation. A common example involves preprocessing digital images in computer vision tasks. 

- 降维或流形学习：将项目的初始表示转换为这些项目的低维表示，同时保留初始表示的一些属性。一个常见的例子涉及计算机视觉任务中的数字图像预处理。

  + manifold 多

  The main practical objectives of machine learning consist of generating accurate predictions for unseen items and of designing eﬃcient and robust algorithms to produce these predictions, even for large-scale problems. To do so, a number of algorithmic and theoretical questions arise. Some fundamental questions include:

  机器学习的主要实践目标包括生成对未知项的精确预测，设计高效和强大的算法来生成这些预测，即使是针对大规模问题。为此，出现了许多算法和理论问题。一些基本问题包括：