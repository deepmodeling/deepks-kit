import os
import numpy as np
from glob import glob


BOHR = 0.52917721092
ELEMENTS = ['X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]
CHARGES = dict(((x,i) for i,x in enumerate(ELEMENTS)))


def parse_xyz(filename):
    with open(filename) as fp:
        natom = int(fp.readline())
        comments = fp.readline().strip()
        atom_str = fp.readlines()
    atom_list = [a.split() for a in atom_str if a.strip()]
    elements = [a[0] for a in atom_list]
    coords = np.array([a[1:] for a in atom_list], dtype=float)
    return natom, comments, elements, coords


def parse_unit(rawunit):
    if isinstance(rawunit, str):
        try:
            unit = float(rawunit)
        except ValueError:
            if rawunit.upper().startswith(('B', 'AU')):
                unit = BOHR
            else: #unit[:3].upper() == 'ANG':
                unit = 1.
    else:
        unit = rawunit
    return unit


def load_array(file):
    ext = os.path.splitext(file)[-1]
    if "npy" in ext:
        return np.load(file)
    elif "npz" in ext:
        raise NotImplementedError
    else:
        try:
            arr = np.loadtxt(file)
        except ValueError:
            arr = np.loadtxt(file, dtype=str)
        return arr


def load_glob(pattern):
    [fn] = glob(pattern)
    return load_array(fn)


def load_system(xyz_file):
    base, ext = os.path.splitext(xyz_file)
    assert ext == '.xyz'
    natom, _, ele, coord = parse_xyz(xyz_file)
    try:
        energy = load_glob(f"{base}.energy*").reshape(1)
    except:
        energy = None
    try:
        force = load_glob(f"{base}.force*").reshape(natom, 3)
    except:
        force = None
    try:
        dm = load_glob(f"{base}.dm*")
        nao = np.sqrt(dm.size).astype(int)
        dm = dm.reshape(nao, nao)
    except:
        dm = None
    return ele, coord, energy, force, dm


def dump_systems(xyz_files, dump_dir, unit="Bohr", ext_type=False):
    print(f"saving to {dump_dir} ... ", end="", flush=True)
    os.makedirs(dump_dir, exist_ok=True)
    if not xyz_files:
        print("empty list! did nothing")
        return
    unit = parse_unit(unit)
    a_ele, a_coord, a_energy, a_force, a_dm = map(np.array,
        zip(*[load_system(fl) for fl in xyz_files]))
    a_coord /= unit
    if ext_type:
        ele = a_ele[0]
        assert all(e == ele for e in a_ele), "element type for each xyz file has to be the same"
        np.savetxt(os.path.join(dump_dir, "type.raw"), ele, fmt="%s")
        np.save(os.path.join(dump_dir, "coord.npy"), a_coord)
    else:
        a_chg = [[[CHARGES[e]] for e in ele] for ele in a_ele]
        a_atom = np.concatenate([a_chg, a_coord], axis=-1)
        np.save(os.path.join(dump_dir, "atom.npy"), a_atom)
    if not all(ene is None for ene in a_energy):
        assert not any(ele is None for ele in a_energy)
        np.save(os.path.join(dump_dir, "energy.npy"), a_energy)
    if not all(ff is None for ff in a_force):
        assert not any(ff is None for ff in a_force)
        a_force *= unit
        np.save(os.path.join(dump_dir, "force.npy"), a_force)
    if not all(dm is None for dm in a_dm):
        assert not any(dm is None for dm in a_dm)
        np.save(os.path.join(dump_dir, "dm.npy"), a_dm)
    print(f"finished", flush=True)
    return


def main(xyz_files, dump_dir=".", group_size=-1, group_prefix="sys", unit="Bohr", ext_type=False):
    if isinstance(xyz_files, str):
        xyz_files = [xyz_files]
    if group_size <= 0:
        dump_systems(xyz_files, dump_dir, unit=unit, ext_type=ext_type)
        return
    ns = len(xyz_files)
    ngroup = np.ceil(ns / group_size).astype(int)
    nd = max(len(str(ngroup)), 2)
    for i in range(ngroup):
        dump_systems(xyz_files[i*group_size:(i+1)*group_size],
                     os.path.join(dump_dir, f"{group_prefix}.{i:0>{nd}d}"),
                     unit=unit, ext_type=ext_type)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="convert .xyz files and corresponding properties "
                    "into systems with .npy files grouped in folders.",
        argument_default=argparse.SUPPRESS)
    parser.add_argument("xyz_files", metavar='FILE', nargs="+", 
                        help="input xyz files")
    parser.add_argument("-d", "--dump-dir", 
                        help="directory of dumped system, default is current dir")
    parser.add_argument("-U", "--unit", 
                        help="length unit used to save npy files (assume xyz in Angstrom)")
    parser.add_argument("-G", "--group-size", type=int, 
                        help="if positive, split data into sub systems with given size, default: -1")
    parser.add_argument("-P", "--group-prefix", 
                        help=r"save sub systems with given prefix as `$dump_dir/$prefix.ii`, default: sys")
    parser.add_argument("-T", "--ext-type", action="store_true", 
                        help="if set, save the element type into separete `type.raw` file")
    args = parser.parse_args()

    main(**vars(args))



