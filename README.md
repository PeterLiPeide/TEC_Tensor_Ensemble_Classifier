# TEC_Tensor_Ensemble_Classifier

This is our updated python package to apply the random projection-based tensor ensemble classifier from our paper "Li, P., Karim, R., & Maiti, T. (2021). TEC: Tensor Ensemble Classifier for Big Data. arXiv preprint arXiv:2103.00025."

We provide a python module which include the CP-STM from "He, L., Kong, X., Yu, P. S., Yang, X., Ragin, A. B., & Hao, Z. (2014, April). Dusk: A dual structure-preserving kernel for supervised tensor learning with applications to neuroimages. In Proceedings of the 2014 SIAM International Conference on Data Mining (pp. 127-135). Society for Industrial and Applied Mathematics.",  the random projection-based CP-STM and TEC model from our paper. The module provides two way to estimate model using either Quadratic programming on Hinge loss, or Gaussian-Newton method on Squared Hinge loss function. 

This repo intends to provide more details about the algorithmtic aspect of our establishment. We include all numerical resutls from our simulation and real data analysis for reproducible purpose. 
