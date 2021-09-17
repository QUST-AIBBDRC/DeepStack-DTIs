##DeepStack-DTIs

DeepStack-DTIs: Predicting drug-target interactions using LightGBM feature selection and deep stacked ensemble classifier.


###DeepStack-DTIs uses the following dependencies:
* MATLAB2014a
* python 3.6 
* numpy
* pandas
* scikit-learn
* imblearn 
* TensorFlow 
* keras

###Guiding principles:

**The dataset file contains the gold standard dataset, Kuang dataset and network dataset.

**feature extraction:
1) Evolutionary-based features: PsePSSM.m is the implementation of PsePSSM. 
2) Sequence-based features: PAAC.py is the implementation of PseAAC.
3) Structural-based features: Structure.py is the implementation of structural information based on SPIDER3.
   
** feature selection:
   GINI.py represents the Gini coefficient.
   IG.py represents the information gain.
   LGBM_selection.py represents the LightGBM feature selection.
   MDS.py represents the locally multidimensional scaling analysis.
   MI.py represents the mutual information.
   MRMD.py represents the max-relevance-max-distance.
   ReliefF.py represents the ReliefF.
   
** data balancing:
   EasyEnsemble.py is the implementation of EasyEnsemble. 
   NearMiss.py is the implementation of NearMiss. 
   RUS.py is the implementation of random undersampling.
   SMOTE.R is the implementation of SMOTE. 

** Classifier:
   AdaBoost.py is the implementation of AdaBoost.
   Deepstack.py is the implementation of deep stacked ensemble classifier. 
   DNN.py is the implementation of deep neural network.
   DT.py is the implementation of decision tree. 
   GRU.py is the implementation of gated recurrent unit. 
   KNN.py is the implementation of K-nearest neighbor. 
   LR.py is the implementation of logistic regression. 
   SVM.py is the implementation of support vector machine.
   XGBoost.py is the implementation of eXtreme gradient boosting. 
