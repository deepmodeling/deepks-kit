Label preperation
=================



System structure file
---------------------

To train a DeePKS model, users must provide the structures of the interested system(s). Structures can be obtained either from a short AIMD run or by adding structural pertubations on top of an optimized geometry.
The structures of the system can be provided via three formats as follows

- **(recommended) grouped into** ``atom.npy``

  The shape of ``atom.npy`` file is **[nframes, natoms, 4]**:
    - **nframes** refers to the number of frames (structures) of the interested system; 
    - **natoms** refers to the number of atoms of the interested system, e.g., for single water system, **natoms = 3**; 
    - the last dimension **4** corresponds to the nuclear charge of the given atom and its *xyz* coordinates.

- **grouped into** ``coord.npy`` **and** ``type.raw``

  ``coord.npy`` is very similar to ``atom.npy`` with the shape **[nframes, natoms, 3]**. The only difference is that the nuclear charge is not included in the last dimension. **Note that this format has not been fully tested for periodic systems.**
  
- **single xyz**
  
  Save the xyz coordinate of each frame as single xyz file, e.g., ``0000.xyz``, ``0001.xyz``,... **Note that this format has not been fully tested for periodic systems.**

It should be noted that if the lattice vectors of each frame are *not* the same, users should specify the lattice vector for each frame via ``box.npy``, of which the shape is **[nframe, 9]**. 
If the prepared structures share the same lattice vector, then users may specify it as a keyword in input files. See xxx for details. 

Property labels
----------------

To train a DeePKS model, the target energy of the interested system is required, and its format should follow the format of the structure file. Additional properties can also be trained, including *force*, *stress*, and *bandgap*. The formats of structure files (taking *atom.npy* as an example) and the corresponding formats of various property labels are summarized as follows:

.. csv-table:: 
   :header: "Filename", "Description", "Shape", "Unit"

   "atom.npy",               "structural file, **required**",      "[nframes, natoms, 4]",  "Bohr or Angstrom"
   "box.npy",               "lattice vector file, optional",      "[nframes, 9]",       "Bohr or Angstrom"
   "energy.npy",              "energy label, **required**",             "[nframes,1]",      "Hartree"
   "force.npy",               "force label, optional",         "[nframes, natoms, 3]",  "Hartree/Bohr"
   "stress.npy",            "virial vector file, optional",      "[nframes, 9]",        "Hartree"
   "orbital.npy",              "bandgap label, optional",             "[nframes,1]",    "Hartree"

