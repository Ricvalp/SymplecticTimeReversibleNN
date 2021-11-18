# SymplecticTimeReversibleNN

### Time-reversible neural networks for learning (Hamiltonian) dynamical systems.

### Four systems are used in the experiments: The Poincare' map of the Henon-Heiles Hamiltonian system for three different values of energy, the Poincare' map of the perturbed simple pendulum, and two 3-dimensional systems that are not Hamiltonian but are reversible with respect to two different reversing symmetries. The repository contains three folders with the experiments, one with the results, and a folder with the methods used in the scripts.

* **AllMethods**

* **PerturbedPendulum**
  + DatasetGenerator.ipynb
  + SympNet.py
  + HenonMaps.py
  + RealNVP.py
  + NeuralNetwork.py

* **HenonHeiles**
  + DatasetGenerator.ipynb
  + SympNet.py
  + HenonMaps.py
  + RealNVP.py
  + NeuralNetwork.py

* **3D**
  + DatasetGenerator.ipynb
  + RealNVP.py
  + NeuralNetwork.py

* **Results**

The scripts SympNets.py, HenonMaps.py, RealNVP.py, and NeuralNetwork.py import .txt files generated by DatasetGenerator.ipynb using the Julia language. The Python scripts do not require Julia.
