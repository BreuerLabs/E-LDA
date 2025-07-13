# Exemplar-LDA


This repository contains Python implementations of **Exemplar-LDA** and its associated algorithms, experiments, baselines, and datasets from the **ICML'25** paper:   



**E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time**

**Link:** [https://arxiv.org/pdf/2506.07747?](https://arxiv.org/pdf/2506.07747?)

**Citation:**

	@inproceedings{breuerlda,
	  title={E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time},
	  author={Breuer, Adam},
	  booktitle={International Conference on Machine Learning},
	  year={2025}
	}



### Algorithms
Our implementations of the main serial (nonparallel) algorithms in the paper are included in the ``/src/Exemplar_LDA.py`` function library. The function ``exemplar_LDA(...)`` implements the following algorithms and calls the associated subroutines described in the paper:

- **FastInitialize-E-LDA**

- **FastGreedy-E-LDA and its associated UpdateHeap subroutine**

### Topic Generators
We include efficient Python implementations of the Topic Generators described in the paper in the ``/src/topic_generators.py`` function library

- **Raw-Umass**

- **Exp-Umass**
 
- **Co-Occurrence**

### Benchmark Datasets
We provide conveniently preprocessed versions (with the code we used to preprocess them) of the three standard public benchmark datasets that are commonly used to benchmark NLP algorithms across the social sciences. These are organized in the respective subdirectories of the ``preprocessed_data`` directory.

- **Congress** - *Matt Thomas, Bo Pang, and Lillian Lee. Get out the vote: determining support or opposi- tion from congressional floor-debate transcripts. In Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing, pages 327–335, 2006.*

- **Reuters** - *David Lewis. Reuters-21578 text categorization test collection. Distribution 1.0, AT&T Labs- Research, 1997.*

- **NewsGroups** - *Ken Lang. Newsweeder: Learning to filter netnews. In Machine learning proceedings 1995, pages 331–339. Elsevier, 1995.*

We also include a library of numerous helper functions (in the ``/src/preprocessing_util.py`` Python file) that efficiently compute normalized UMass coherence (used to replicate the coherence plots for Experiments Set 2 in Section 9 of the paper), as well as many text preprocessing helper functions. 

We also include **Gensim's MALLET Python wrapper file**, which has been updated to run with new NumPy, etc. libraries on ARM chips. 

To run LDA baselines, you will need an installation of **Java** and the **MALLET** Java topic modeling library (*Andrew Kachites McCallum. Mallet: A machine learning for languagetoolkit.*) For details, see *https://mimno.github.io/Mallet/topics.html*

To run neural and LLM-based baselines (*BERTopic, AVITM, ETM*), you will also need the authors' code libraries. For BERTopic, install the author's *bertopic* Python package (*https://maartengr.github.io/BERTopic/index.html*). For *AVITM* and *ETM*, download the most up-to-date repos of the respective authors' codebases, which are now part of the *OCTIS* library (*https://github.com/MIND-Lab/OCTIS*). To run these baselines via *OCTIS*, place our *RUN_baselines_ETM_AVITM.py* script and *preprocessed_data/* directory into the *OCTIS-master/* main directory and place our source files from our *src/* directory into the OCTIS *OCTIS-master/src/* directory.