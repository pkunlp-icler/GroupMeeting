# GroupMeeting

* [Tasks or Methods](#tasks-or-methods)
    * [Relation Extraction](#1-relation-extraction)
    * [Semantic Role Labeling](#2-semantic-role-labeling)
    * [Dependency Parsing](#3-dependency-parsing)
    * [Math Problem Generation](#4-math-problem-generation)
    * [Medical Text Processing](#5-medical-text-processing)
    * [Named Entity Recognition](#6-named-entity-recognition)
    * [Law Judgement Prediction](#7-law-judgement-prediction)
    * [Discourse Parsing](#8-discourse-parsing)
    * [Knowledge Distilling](#9-knowledge-distilling)
    * [Data Augmentation](#10-data-augmentation)
    * [Transformer](#11-transformer)
    * [Numerical Embedding](#12-numerical-embedding)
    * [Question Answering](#13-question-answering)
* [Datasets](#datasets)
   

## Tasks or Methods
### 1. Relation Extraction
2020/02/23
reporter: Shuang Zeng
* paper: Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou. 2019. [Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs](https://arxiv.org/abs/1909.00228). In *EMNLP-IJCNLP 2019*.
* ppt:  [20200223_zs](ppts/20200223_zs.pdf)
* method: BiLSTM + Edge-oriented Graph + Iterative Inference Mechanism

2020/03/08
reporter: Yuan Zong
* paper: Daojian Zeng, Haoran Zhang, Qianying Liu. 2020. [CopyMTL:Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning](https://arxiv.org/abs/1911.10438). In *AAAI 2020*.
* ppt:  [20200308_zy](ppts/20200308_zy.pdf)
* method: Pointer-generator + mulit-task

2020/03/15
reporter: Runxin Xu
* paper: Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, Yi Chang. 2019. [A Novel Hierarchical Binary Tagging Framework for Joint Extraction of Entities and Relations](https://arxiv.org/abs/1909.03227). In *arxiv preprint*.
* ppt:  [20200315_xrx](ppts/20200315_xrx.pdf)
* method: BERT + subject tagger + relation-specific object tagger

2020/03/22
reporter: Shuang Zeng
* paper: Bowen Yu, Zhenyu Zhang, Tingwen Liu, Bin Wang, Sujian Li, Quangang Li. 2019. [Beyond Word Attention: Using Segment Attention in Neural Relation Extraction](https://www.ijcai.org/Proceedings/2019/750). In *IJCAI 2019*.
* ppt:  [20200322_zs](ppts/20200322_zs.pdf)
* method: Segment Attention Layer with linear-chain CRF + two regularizers

2020/04/05
reporter: Yuan Zong
* paper: Christoph Alt, Marc Hübner, Leonhard Hennig. 2019. [Fine-tuning Pre-Trained Transformer Language Models to Distantly Supervised Relation Extraction](https://arxiv.org/abs/1906.08646). In *ACL 2019*.
* ppt:  [20200405_zy](ppts/20200405_zy.pdf)
* method: Pretrain_model + bag_level

2020/04/12
reporter: Shuang Zeng
* paper1: Xiaocheng Feng, Jiang Guo, Bing Qin, Ting Liu, Yongjie Liu. 2017. [Effective Deep Memory Networks for Distant Supervised Relation Extraction](https://www.ijcai.org/Proceedings/2017/559). In *IJCAI 2017*.
* method: PCNN + word-level memory + sentence-level memory, bag-level multi-instance multi-label learning
* paper2: Jianhao Yan, Lin He, Ruqin Huang, Jian Li, Ying Liu. 2019. [Relation Extraction with Temporal Reasoning Based on Memory Augmented Distant Supervision](https://www.aclweb.org/anthology/N19-1107). In *NAACL 2019*.
* method: PCNN + temporal memory + iterative reasoning
* ppt:  [20200412_zs](ppts/20200412_zs.pdf)

2020/04/20
reporter: Xudong Chen
* paper: Yang Li, Guodong Long, Tao Shen, Tianyi Zhou, Lina Yao, Huan Huo, Jing Jiang.2020. [Self-Attention Enhanced Selective Gate with Entity-Aware Embedding for Distantly Supervised Relation Extraction](https://arxiv.org/abs/1911.11899?context=cs). In *AAAI 2020*.
* ppt:  [20200420_cxd](ppts/20200420_cxd.pdf)
* method: Attention+Gate+Entity-Aware Embedding

2020/04/20
reporter: Runxin Xu
* paper1: Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, Tom Kwiatkowski.2019. [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158). In *ACL 2019*.
* paper2: Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang.2020. [K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606). In *AAAI 2020*.
* ppt:  [20200420_xrx](ppts/20200420_xrx.pdf)
* method: Pre-train Model & Knowledge Graph

2020/05/03
reporter: Yuan Zong
* paper: Wenya Wang, Sinno Jialin Pan. 2020. [Integrating Deep Learning with Logic Fusion for Information Extraction
](https://arxiv.org/abs/1912.03041). In *AAAI 2020*.
* ppt:  [20200503_zy](ppts/20200503_zy.pdf)
* method: Transformer & Logic Fusion

2020/05/17
reporter: Shuang Zeng
* paper: Kai Sun, Richong Zhang, Samuel Mensah, Yongyi Mao, Xudong Liu. 2020. [Recurrent Interaction Network for Jointly Extracting Entities and Classifying Relations](https://arxiv.org/abs/2005.00162). In *Arxiv Preprint*.
* ppt:  [20200517_zs](ppts/20200517_zs.pdf)
* method: Recurrent Interaction Network

2020/06/14
reporter: Shuang Zeng
* paper: Guoshun Nan, Zhijiang Guo, Ivan Sekulić, Wei Lu. 2020. [Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312). In *ACL 2020*.
* ppt:  [20200614_zs](ppts/20200614_zs.pdf)
* method: Node Constructor + Structure Attention based Sturcture Induction + Multi-hop Reasoning


### 2. Semantic Role Labeling
2020/02/23
reporter: Xudong Chen
* paper: Zuchao Li, Shexia He, Junru Zhou, Hai Zhao, Kevin Par Rui Wang. 2019. [Dependency and Span, Cross-Style Semantic Role Labeling on PropBank and NomBank](https://arxiv.org/abs/1911.02851). In *arxiv preprint*.
* ppt:  [20200223_cxd](ppts/20200223_cxd.pdf)
* method: BiLSTM + Syntactic Aided + Biaffine

2020/03/22
reporter: Xudong Chen
* paper: Anwen Hu,Zhicheng Dou,Jian-Yun Nie,Ji-Rong Wen. 2019. [Leveraging Multi-token Entities in Document-level Named Entity Recognition](http://playbigdata.ruc.edu.cn/dou/publication/2020_aaai_ner.pdf). In *AAAI 2020*.
* ppt:  [20200322_cxd](ppts/20200322_cxd.pdf)
* method: BiLSTM + Document Attention + CRF


### 3. Dependency Parsing
2020/03/15
reporter: Wei Cui
* paper: Daniel Fern´andez-Gonz´alez and Carlos G´omez-Rodr´ıguez. 2020. [Discontinuous Constituent Parsing with Pointer Networks](https://arxiv.org/abs/2002.01824). In *AAAI 2020*.
* ppt:  [20200315_cw](ppts/20200315_cw.pdf)
* method: Constituents as Augmented Dependencies + Pointer Network

2020/04/12
reporter: Wei Cui
* paper: Alireza Mohammadshahi and James Henderson. 2020. [Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement](https://arxiv.org/abs/2003.13118). In *arxiv preprint*.
* ppt:  [20200412_cw](ppts/20200412_cw.pdf)
* method: Graph-to-Graph Iterative Refinement


### 4. Math Problem Generation
2020/03/01
reporter: Tianyang Cao 
* paper: Xinyu Hua and Lu Wang. 2019. [Sentence-Level Content Planning and Style Specification for Neural Text Generation](https://arxiv.org/abs/1909.09734). In *arxiv preprint*.
* ppt: [20200301_cty](ppts/20200301_cty.pdf)
* method: context planning decoder+ style specification

2020/03/08
reporter: Songge Zhao
* paper: Cao Liu, Kang Liu .2019. [Generating Questions for Knowledge Bases via Incorporating Diversified Contexts and Answer-Aware Loss](https://arxiv.org/abs/1910.13108). In *EMNLP 2019*.
* ppt: [20200308_zsg](ppts/20200308_zsg.pdf)
* method: Divisified context + Transformer + Answer-aware loss

2020/03/29
repoter: Tianyang Cao
* paper: Deng Cai, Yan Wang, Wei Bi, ZhaoPeng Tu, XiaoJiang Liu, Shuming Shi. 2019. [Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework](https://www.aclweb.org/anthology/D19-1195.pdf). In *EMNLP 2019*.
* ppt: [20200329_cty](ppts/20200329_cty.pdf)
* method: Matching score + skeleton to generation


2020/04/05
repoter: Songge Zhao
* paper: Weichao Wang , Shi Feng , Daling Wang , Yifei Zhang [Answer-guided and Semantic Coherent Question Generation in Open-domain Conversation](https://www.aclweb.org/anthology/D19-1511/). In *EMNLP 2019*.
* ppt: [20200405_zsg](ppts/20200405_zsg.pdf)
* method:RL + AL + CVAE genration

2020/04/26
repoter:Tianyang Cao
* paper: Yun-Zhu Song, Hong-Han Shuai, Sung-Lin Yeh,  Yi-Lun Wu, Lun-Wei Ku, Wen-Chih Peng. 2020. [Attractive or Faithful? Popularity-Reinforced Learning for Inspired Headline Generation](https://arxiv.org/pdf/2002.02095.pdf). In *AAAI 2020*
* ppt: [20200426_cty](ppts/20200426_cty.pdf)
* method: Article extractor with popular topic attention + popularity predictor + RL

2020/05/24
repoter: Tianyang Cao
* paper: Siqi Bao, Huang He, Fan Wang, Hua Wu and Haifeng Wang. 2020. [PLATO: Pre-trained Dialogue Generation Model with
Discrete Latent Variable](https://arxiv.org/pdf/1910.07931.pdf). In *ACL 2020* 
* ppt: [20200524_cty](ppts/20200524_cty.pdf)
* method: Discrete latent variable + context-response matching pretraining

2020/06/21
repoter: Tianyang Cao
* paper: Di Jin, Zhijing Jin, Joey Tianyi Zhou,Lisa Orii,Peter Szolovits. 2020. [Hooks in the Headline: Learning to Generate Headlines with Controlled Styles](http://xxx.itp.ac.cn/pdf/2004.01980.pdf). In *ACL 2020*
* ppt: [20200621_cty](ppts/20200621_cty.pdf)
* method: Multitask learning and parameters sharing 

2020/06/28
repoter: Songge Zhao
* paper: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah. 2020. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). 
* ppt: [20200628_zsg](ppts/20200628_zsg.pdf)
* method: GPT3 pre_train LM

2020/07/12
repoter: Tianyang Cao
* paper: Zhiliang Tian, Wei Bi, Dongkyu Lee, Lanqing Xue, Yiping Song, Xiaojiang Liu,  Nevin L. Zhang. 2020.  [Response-Anticipated Memory for
On-Demand Knowledge Integration in Response Generation](https://www.aclweb.org/anthology/2020.acl-main.61.pdf)
* ppt: [20200712_cty](ppts/20200712_cty.pdf)
* method: teacher-student framework + response-aware document memory construction


### 5. Medical Text Processing
2020/03/01
reporter: Huan Zhang
* paper: Bansal, T., Verga, P., Choudhary, N., & McCallum, A. 2019. [Simultaneously Linking Entities and Extracting Relations from Biomedical Text Without Mention-level Supervision](https://arxiv.org/abs/1912.01070). In *AAAI 2020*.
* ppt:  [20200301_zh](ppts/20200301_zh.pdf)
* method: Simultaneously link entities and extract relationships

### 6. Named Entity Recognition
2020/03/29
reporter: Huan Zhang
* paper: Angli Liu, Jingfei Du, Veselin Stoyanov,  2019. [Knowledge-Augmented Language Model and Its Application to Unsupervised Named-Entity Recognition](https://arxiv.org/abs/1904.04458). In *NAACL 2019*.
* ppt:  [20200329_zh](ppts/20200329_zh.pdf)
* method: LM + KB + unsupervised + ner(plain text)

2020/04/26
reporter: Huan Zhang
* paper: Changmeng Zheng, Yi Cai, Jingyun Xu, Ho-fung Leung and Guandong Xu, 2019. [A Boundary-aware Neural Model for Nested Named Entity Recognition](https://www.aclweb.org/anthology/D19-1034.pdf). In *EMNLP 2019*.
* ppt:  [20200426_zh](ppts/20200426_zh.pdf)
* method: Boundary-aware + categorical labels prediction + nested ner

2020/05/24
reporter: Huan Zhang
* paper: Esteban Safranchik, Shiying Luo, Stephen H. Bach, 2020. [Weakly Supervised Sequence Tagging from Noisy Rules](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf). In *AAAI 2020*.
* ppt:  [20200524_zh](ppts/20200524_zh.pdf)
* method: Weakly Supervised + Rules

2020/06/14
reporter: Xudong Chen
* paper: Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li, 2020. [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/pdf/1910.11476.pdf). In *ACL 2020*.
* ppt:  [20200614_cxd](ppts/20200614_cxd.pdf)
* method: Weakly Supervised + Rules

2020/06/21
reporter: Huan Zhang
* paper: Hiroki Ouchi, Jun Suzuki, Sosuke Kobayashi, Sho Yokoi, Tatsuki Kuribayashi, Ryuto Konno, Kentaro Inui, 2020. [Instance-Based Learning of Span Representations:A Case Study through Named Entity Recognition](https://arxiv.org/pdf/2004.14514.pdf). In *ACL 2020*.
* ppt:  [20200621_zh](ppts/20200621_zh.pdf)
* method: instence-based + ner

2020/06/28
reporter: Yuan Zong
* paper: Xiang Dai, Sarvnaz Karimi, Ben Hachey, Cecile Paris1 2020. [An Effective Transition-based Model for Discontinuous NER](https://arxiv.org/abs/2004.13454). In *ACL 2020*.
* ppt:  [20200628_zy](ppts/20200628_zy.pdf)
* method: ner like dependency parsing


2020/07/12
reporter: Huan Zhang
* paper: Bill Yuchen Lin, Dong-Ho Lee, Ming Shen, Ryan Moreno, Xiao Huang, Prashant Shiralkar, Xiang Ren, 2020. [TriggerNER: Learning with Entity Triggers as Explanations for Named Entity Recognition](https://arxiv.org/pdf/2004.07493v1.pdf). In *ACL 2020*.
* ppt:  [20200712_zh](ppts/20200712_zh.pdf)
* method: entity trigger + ner -> cost-effective result


### 7. Law Judgement Prediction
2020/05/03
reporter: Xudong Chen
* paper: Nuo Xu, PinghuiWang, Long Chen, Li Pan, Xiaoyan Wang, Junzhou Zhao,  2020. [Distinguish Confusing Law Articles for Legal Judgment Prediction](https://arxiv.org/pdf/2004.02557.pdf). In *ACL 2020*.
* ppt:  [20200503_cxd](ppts/20200503_cxd.pdf)
* method: GCL + GDO

### 8. Discourse Parsing
2020/05/12
reporter: Wei Cui
* paper: Longyin Zhang, Yuqing Xing, Fang Kong, Peifeng Li, Guodong Zhou. 2020. [A Top-Down Neural Architecture towards Text-Level Parsing of Discourse Rhetorical Structure](https://arxiv.org/abs/2005.02680). In *ACL 2020*.
* ppt:  [20200512_cw](ppts/20200512_cw.pdf)
* method: GRU-CNN-GRU encoder + biaffine attention + Attention-based Encoder-Decoder on Split Point Ranking


### 9. Knowledge Distilling
2020/05/12
reporter: Runxin Xu
* paper: Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. 2020. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). In *ICLR 2020*.
* ppt:  [20200512_xrx](ppts/20200512_xrx.pdf)
* method: Konwledge Distilling


### 10. Data Augmentation
2020/05/31
reporter: Yuan Zong
* paper: Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. 2020. [MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classiﬁcation](https://arxiv.org/abs/2004.12239). In *ACL 2020*.
* ppt:  [20200531_zy](ppts/20200531_zy.pdf)
* method: Data mixing + Semi-supervised

### 11. Transformer
2020/06/07
reporter: Runxin Xu
* paper: Hao Peng, Roy Schwartz, Dianqi Li, Noah A. Smith. 2020. [A Mixture of h−1 Heads is Better than h Heads](https://arxiv.org/abs/2005.06537). In *ACL 2020*.
* ppt:  [20200607_xrx](ppts/20200607_xrx.pdf)
* method: Consider multi-head self-attention as a mixture of experts + modify gate mechanism


### 12. Numerical Embedding
2020/05/17
reporter: Songge Zhao
* paper: Mor Geva, Ankit Gupta, Jonathan Berant. 2020. [Injecting Numerical Reasoning Skills into Language Models](https://arxiv.org/abs/2004.04487). In *ACL 2020*.
* ppt:  [20200517_zsg](ppts/20200517_zsg.pdf)
* method: Specific Pre-training + fine-tuning

2020/05/31
reporter: Songge Zhao
* paper: Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, Matt Gardner. 2019. [Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://arxiv.org/abs/1909.07940). In *EMNLP 2019*.
* ppt:  [20200531_zsg](ppts/20200531_zsg.pdf)
* method: to be fixed

### 13. Question Answering
2020/07/05
reporter: Shuang Zeng
* paper1: Lin Qiu, Yunxuan Xiao, Yanru Qu, Hao Zhou, Lei Li, Weinan Zhang, Yong Yu. 2019. [Dynamically Fused Graph Network for Multi-hop Reasoning](https://www.aclweb.org/anthology/P19-1617/). In *ACL 2019*.
* paper2: Nan Shao, Yiming Cui, Ting Liu, Shijin Wang, Guoping Hu. 2020. [Is Graph Structure Necessary for Multi-hop Reasoning?](https://arxiv.org/abs/2004.03096). In *Arxiv Preprint*.
* ppt:  [20200705_zs](ppts/20200705_zs.pdf)
* method: Paragraph Selector + Constructing Entity Graph + Encoding Query and Context + Reasoning with the Fusion Block

2020/07/05
reporter: Xudong Chen
* paper: Shaoru Guo, Ru Li, Hongye Tan, Xiaoli Li, Yong Guan,Hongyan Zhao and Yueping Zhang. 2020. [A Frame-based Sentence Representation for Machine Reading
Comprehension](https://www.aclweb.org/anthology/2020.acl-main.83.pdf). In *ACL 2020*.
* ppt:  [20200705_cxd](ppts/20200705_cxd.pdf)
* method: Bert + Frame-based Sentence Representation

### 14. Chinese Word Segmentation
reporter: Yuan Zong
* paper: Ning Ding, Dingkun Long, Guangwei Xu, Muhua Zhu, Pengjun Xie, Xiaobin Wang, Hai-Tao Zheng. 2020. [Coupling Distant Annotation and Adversarial Training for Cross-Domain Chinese Word Segmentation](https://arxiv.org/pdf/2007.08186.pdf). In *ACL 2020*.
* ppt:  [20200717_zy](ppts/20200719_zy.pdf)
* method: Distant Annotation and Adversarial Training

## Datasets
