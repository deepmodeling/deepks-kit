Getting Started
================

Label preperation
-----------------
To train a DeePKS model, users must provide the structure and the corresponding target energy of the interested system(s). 
The structure of the system can be provided via three formats as follows

- **(recommended) grouped into** *atom.npy*

  The shape of *atom.npy* file is *nframes x natoms x 4*:
    - *nframes* refers to the number of frames (structures) of the interested system; 
    - *natoms* refers to the number of atoms of the interested system, e.g., for single water system, *natoms = 3*; 
    - the last dimension *4* corresponds to the nuclear charge of the given atom and its *xyz* coordinates.

- **grouped into** *coord.npy* **and** *type.raw*

  - *coord.npy* is very similar to *atom.npy* with the shape *nframes x natoms x 3*. The only difference is that the nuclear charge is not included in the last dimension.
  
- **single xyz**

Input file preperation
----------------------
