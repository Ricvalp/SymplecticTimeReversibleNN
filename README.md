# SymplecticTimeReversibleNN

### Time-reversible neural networks for learning (Hamiltonian) dynamical systems.

### Two systems are used in the experiments: The Poincare' map of the Henon-Heiles Hamiltonian system for E=0.1, and the Poincare' map of the perturbed simple pendulum. The repository contains 

* **AllMethods** Library of methods

* **PerturbedPendulum**
  + *DatasetGenerator.ipynb* which generates the dataset. It needs Julia, but the .txt files are already in the repository
  + *Notebook_HM.ipynp* which implements the space of time-reversible compositions of Henon maps
  + *Notebook_R.ipynp* which implements the space of time-reversible compositions of Real NVPs
  + *Notebook_SN.ipynp* which implements the space of SympNets from [Jin et al. (2020)](https://arxiv.org/abs/2001.03750)
  + *Notebook_NN.ipynp* which implements the space of multi layer perceptrons

* **HenonHeiles**
  + *DatasetGenerator.ipynb* which generates the dataset. It needs Julia, but the .txt files are already in the repository
  + *Notebook_HM.ipynp* which implements the space of time-reversible compositions of Henon maps
  + *Notebook_R.ipynp* which implements the space of time-reversible compositions of Real NVPs
  + *Notebook_SN.ipynp* which implements the space of SympNets from [Jin et al. (2020)](https://arxiv.org/abs/2001.03750)
  + *Notebook_NN.ipynp* which implements the space of multi layer perceptrons
