# DeePQC

DeePQC is a program to generate accurate energy functionals for quantum chemistry systems,
for both post-Hartree-Fock (perturbation) scheme (DeePHF) and self-consistent scheme (DeePKS).

The program contains five sub-commands, 
- `train`: train an neural network based post-HF energy functional model
- `test`: test the post-HF model with given data and show statistics
- `scf`: run self-consistent field calculation with given energy model
- `stat`: collect and print statistics of the SCF the results
- `iterate`: iteratively train an self-consistent model by combining four commands above
