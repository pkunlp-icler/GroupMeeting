# GroupMeeting

* [Task](#task)
    * [Relation Extraction](#1-relation-extraction)
    * [Semantic Role Labeling](#2-semantic-role-labeling)
    * [Dependency Parsing](#3-dependency-parsing)
    * [Math Problem Generation](#4-math-problem-generation)
    * [Medical Text Processing](#5-medical-text-processing)
* [Datasets](#datasets)
   

## Task
### 1. Relation Extraction
2020/02/23
reporter: Zeng Shuang
* paper: Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou. 2019. [Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs](https://arxiv.org/abs/1909.00228). In *EMNLP-IJCNLP 2019*.
* ppt:  [20200223_zengshuang](ppts/20200223_zengshuang.pdf)
* method: BERT + subject tagger + relation-specific object tagger


2020/03/08
reporter: Zong Yuan
* paper: DaojianZeng, HaoranZhang, QianyingLiu. 2020. [CopyMTL:Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning](https://arxiv.org/abs/1911.10438). In *AAAI 20*.
* ppt:  [20200308](ppts/20200308.pdf)
* method: Pointer-generator + mulit-task


2020/03/15
reporter: Runxin Xu
* paper: Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, Yi Chang. 2019. [A Novel Hierarchical Binary Tagging Framework for Joint Extraction of Entities and Relations](https://arxiv.org/abs/1909.03227). In *arxiv preprint*.
* ppt:  [20200315_xrx](ppts/20200315_xrx.pdf)
* method: BiLSTM + Edge-oriented Graph + Iterative Inference Mechanism


### 2. Semantic Role Labeling
2020/02/23
reporter: Chen Xudong
* paper: Zuchao Li, Shexia He, Junru Zhou, Hai Zhao, Kevin Par Rui Wang. 2019. [Dependency and Span, Cross-Style Semantic Role Labeling on PropBank and NomBank](https://arxiv.org/abs/1911.02851). In *arxiv preprint*
* ppt:  [presentaion2.23_cxd.pdf](ppts/presentaion2.23_cxd.pdf)
* method: BiLSTM+Syntactic Aided+Biaffine


### 3. Dependency Parsing
2020/03/15
reporter: Cui Wei
* paper: Daniel Fern´andez-Gonz´alez and Carlos G´omez-Rodr´ıguez. 2020. [Discontinuous Constituent Parsing with Pointer Networks](https://arxiv.org/abs/2002.01824). In *AAAI-2020*.
* ppt:  [20200315_cuiwei](ppts/20200315_cuiwei.pdf)
* method: Constituents as Augmented Dependencies + Pointer Network

### 4. Math Problem Generation
2020/03/01
reporter: Cao Tianyang
* paper: Xinyu Hua and Lu Wang. 2019. [Sentence-Level Content Planning and Style Specification for Neural Text Generation](https://arxiv.org/abs/1909.09734). In *arxiv preprint*
* ppt: [3.1_caotianyang_presentation.pdf](ppts/3.1_caotianyang_presentation.pdf)
* method: context planning decoder+ style specification

2020/03/08
reporter: Zhao Songge
* paper: Cao Liu Kang Liu .2019 [Generating Questions for Knowledge Bases via Incorporating Diversified Contexts and Answer-Aware Loss](https://arxiv.org/abs/1910.13108). In *EMNLP-2019*.
* ppt: [3.8 zhaosongge_presentation.pdf](ppts/3.8_zhaosongge.pdf)
* method: Divisified context + Transformer + Answer-aware loss


### 5. Medical Text Processing
2020/03/01
reporter: Zhang Huan
* paper: Bansal, T., Verga, P., Choudhary, N., & McCallum, A. 2019. [Simultaneously Linking Entities and Extracting Relations from Biomedical Text Without Mention-level Supervision](https://arxiv.org/abs/1912.01070). In *AAAI-2020*.
* ppt:  [20200301_zhanghuan](ppts/20200301_zhanghuan.pdf)
* method: Simultaneously link entities and extract relationships


## Datasets
