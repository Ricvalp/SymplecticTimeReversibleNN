# SymplecticTimeReversibleNN

### Time-reversible neural networks for learning (Hamiltonian) dynamical systems.

### Four systems are used in the experiments: The Poincare' map of the Henon-Heiles Hamiltonian system for three different values of energy, the Poincare' map of the perturbed simple pendulum, and two 3-dimensional, systems that are not Hamiltonian but are reversible with respect to two different reversing symmetries. The repository contains three folders with the experiments.

* **PerturbedPendulum**
  + DatasetGenerator.i
  + SympNet.py
  + HenonMaps.py
  + RealNVP.py
  + NeuralNetwork.py

* **HenonHeiles**
  + DatasetGenerator.i
  + SympNet.py
  + HenonMaps.py
  + RealNVP.py
  + NeuralNetwork.py

* **3D**
  + DatasetGenerator.i
  + RealNVP.py
  + NeuralNetwork.py

* **Results**

In each folder there is a dataset generator (Jupyter Notebook) that generates .txt files using the Julia language. The dataset are then imported and used in the Python scripts that do not require Julia.
