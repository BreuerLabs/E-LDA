# Exemplar-LDA


This repository contains Python implementations of **Exemplar-LDA** and its associated algorithms, experiments, baselines, and datasets. 

### Quick Start
We include two quickstart demo scripts: 

- ``DEMO_CLEAN_PREPARE_TEXT.py`` *downloads a demo dataset, removes stopwords/punctuation, and builds the doc-word matrix;*

- ``DEMO_RUN_E-LDA.py`` *estimates an E-LDA model, analyzes results, and saves them in convenient CSVs.*

**To run E-LDA on your own text dataset**, then you can **either** replace the ``docs_raw`` object in ``DEMO_CLEAN_PREPARE_TEXT.py`` with your own list of strings (each string is the text of one of your documents). **Alternatively, if you have already cleaned your texts** and removed stopwords and punctuation, etc., then you can **skip loading many text cleaning packages** and start with the line ``build_docword_matrix(docs)`` in ``DEMO_CLEAN_PREPARE_TEXT.py``, where ``docs`` is a list of strings (each string is a cleaned text of one of your documents). After ``DEMO_CLEAN_PREPARE_TEXT.py`` has completed, then run ``DEMO_RUN_E-LDA.py`` to load the saved doc-word matrix and estimate an E-LDA model.





### Algorithms
Our implementations of the main algorithms in the paper are included in the ``/src/Exemplar_LDA.py`` function library. The function ``exemplar_LDA(...)`` implements the following algorithms and calls the associated subroutines described in the paper:

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

We also include a library of numerous helper functions (in the ``/src/text_cleaning_util.py``, ``/src/save_load_docs_util.py``, ``/src/ELDA_matrix_building_util.py``, and ``src/coherence.py`` Python files) that efficiently compute document-word matrices (in an efficient index format) and normalized UMass coherence (used to replicate the coherence plots for Experiments Set 2 in Section 9 of the paper), as well as many text preprocessing helper functions. 

We also include **Gensim's MALLET Python wrapper file**, which has been updated to run with new NumPy, etc. libraries on ARM chips. 

To run baselines, you will need an installation of **Java** and the **MALLET** Java topic modeling library (*Andrew Kachites McCallum. Mallet: A machine learning for languagetoolkit.*) For details, see **https://mimno.github.io/Mallet/topics.html**
