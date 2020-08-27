import os
import sys
import glob
import numpy as np
import shutil
import argparse
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.utils import check_list, check_array
from deepqc.utils import load_array, load_yaml, get_with_prefix


def concat_data(systems=None, sys_dir=".", dump_dir=".", pattern="*"):
    if systems is None:
        systems = sorted(filter(os.path.isdir, map(os.path.abspath, glob.glob(f"{sys_dir}/*"))))
    npy_names = list(map(os.path.basename, glob.glob(f"{systems[0]}/*.npy")))
    os.makedirs(dump_dir, exist_ok=True)
    for nm in npy_names:
        tmp_array = np.concatenate([np.load(f"{sys}/{nm}") for sys in systems])
        np.save(f"{dump_dir}/{nm}", tmp_array)
    if os.path.exists(f'{systems[0]}/system.raw'):
        shutil.copy(f'{systems[0]}/system.raw', dump_dir)


def print_stat(systems=None, test_sys=None, 
               dump_dir=None, test_dump=None, group=False,
               with_conv=True, with_e=True, e_name="e_cf", 
               with_f=True, f_name="f_cf"):
    load_func = load_stat if not group else load_stat_grouped
    if dump_dir is None:
        dump_dir = "."
    if test_dump is None:
        test_dump = dump_dir
    if systems is not None:
        tr_c, tr_e, tr_f = load_func(systems, dump_dir, with_conv, 
                                     with_e, e_name, with_f, f_name)
        print("Training:")
        if with_conv:
            print_stat_conv(tr_c, indent=2)
        if with_e:
            shift = tr_e.mean()
            print_stat_e(tr_e, shift=shift, indent=2)
        if with_f:
            print_stat_f(tr_f, indent=2)
    if test_sys is not None:
        ts_c, ts_e, ts_f = load_func(test_sys, test_dump, with_conv, 
                                     with_e, e_name, with_f, f_name)
        print("Testing:")
        if with_conv:
            print_stat_conv(ts_c, indent=2)
        if with_e:
            if not systems: shift = None
            print_stat_e(ts_e, shift=shift, indent=2)
        if with_f:
            print_stat_f(ts_f, indent=2)
    

def print_stat_conv(conv, indent=0):
    nsys = conv.shape[0]
    ind = " "*indent
    print(ind+f'Convergence:')
    print(ind+f'  {np.sum(conv)} / {nsys} = \t {np.mean(conv):.5f}')


def print_stat_e(e_err, shift=None, indent=0):
    ind = " "*indent
    print(ind+"Energy:")
    print(ind+f'  ME: \t {e_err.mean()}')
    print(ind+f'  MAE: \t {np.abs(e_err).mean()}')
    if shift:
        print(ind+f'  MARE: \t {np.abs(e_err-shift).mean()}')


def print_stat_f(f_err, indent=0):
    ind = " "*indent
    print(ind+"Force:")
    print(ind+f'  MAE: \t {np.abs(f_err).mean()}')


def load_stat(systems, dump_dir,
              with_conv=True, with_e=True, e_name="e_cf", 
              with_f=True, f_name="f_cf"):
    systems = check_list(systems)
    c_res = []
    e_err = []
    f_err = []
    for fl in systems:
        lbase = fl.rstrip(os.path.sep).rstrip(".xyz")
        rbase = os.path.join(dump_dir, os.path.basename(lbase))
        if with_conv:
            try:
                c_res.append(load_array(get_with_prefix("conv", rbase, ".npy")))
            except FileNotFoundError as e:
                print("Warning! conv.npy not found:", e, file=sys.stderr)
        if with_e:
            try:
                le = get_with_prefix("energy", lbase, ".npy")
                re = get_with_prefix(e_name, rbase, ".npy")
                e_err.append(load_array(le) - load_array(re))
            except FileNotFoundError as e:
                print("Warning! energy file not found:", e, file=sys.stderr)
        if with_f:
            try:
                lf = get_with_prefix("force", lbase, ".npy")
                rf = get_with_prefix(f_name, rbase, ".npy")
                f_err.append(load_array(lf) - load_array(rf))
            except FileNotFoundError as e:
                print("Warning! force file not found:", e, file=sys.stderr)
    return np.concatenate(c_res, 0) if c_res else None, \
           np.concatenate(e_err, 0) if e_err else None, \
           np.concatenate(f_err, 0) if f_err else None


def load_stat_grouped(systems, dump_dir=".",
                      with_conv=True, with_e=True, e_name="e_cf", 
                      with_f=True, f_name="f_cf"):
    systems = check_list(systems)
    lbases = [fl.rstrip(os.path.sep).rstrip(".xyz") for fl in systems]
    if with_conv:
        c_res = load_array(get_with_prefix("conv", dump_dir, ".npy"))
    if with_e:
        e_res = load_array(get_with_prefix(e_name, dump_dir, ".npy"))
        e_lbl = np.concatenate([
            load_array(get_with_prefix("energy", lb, ".npy")) for lb in lbases
        ], 0)
        e_err = e_lbl - e_res
    if with_f:
        f_res = load_array(get_with_prefix(f_name, dump_dir, ".npy"))
        f_lbl = np.concatenate([
            load_array(get_with_prefix("force", lb, ".npy")) for lb in lbases
        ], 0)
        f_err = f_lbl - f_res
    return c_res if with_conv else None, \
           e_err if with_e else None, \
           f_err if with_f else None


def cli():
    parser = argparse.ArgumentParser(
                description="Print the stat of SCF results",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file used for SCF calculation')
    parser.add_argument("-s", "--systems", nargs="*",
                        help='system paths used as training set (i.e. calculate shift)')
    parser.add_argument("-d", "--dump-dir",
                        help="directory used to save SCF results of training systems")
    parser.add_argument("-ts", "--test-sys", nargs="*",
                        help='system paths used as testing set (i.e. not calculate shift)')
    parser.add_argument("-td", "--test-dump",
                        help="directory used to save SCF results of testing systems")
    parser.add_argument("-G", "--group", action='store_true',
                        help="if set, assume results are grouped")
    parser.add_argument("-NC", action="store_false", dest="with_conv",
                        help="do not print convergence results")
    parser.add_argument("-NE", action="store_false", dest="with_e",
                        help="do not print energy results")
    parser.add_argument("-NF", action="store_false", dest="with_f",
                        help="do not print force results")
    parser.add_argument("--e-name",
                        help="name of the energy file (no extension)")
    parser.add_argument("--f-name",
                        help="name of the force file (no extension)")
    args = parser.parse_args()

    if hasattr(args, "input"):
        rawdict = load_yaml(args.input)
        del args.input
        argdict = {fd: rawdict[fd]
                     for fd in ("systems", "dump_dir", "group")
                     if fd in rawdict}
        argdict.update(vars(args))
    else:
        argdict = vars(args)

    print_stat(**argdict)


# Below are legacy tools, kept for old examples

def print_stat_per_sys(err, conv=None, train_idx=None, test_idx=None):
    err = np.array(err).reshape(-1)
    nsys = err.shape[0]
    if conv is not None:
        assert len(conv) == nsys
        print(f'converged calculation: {np.sum(conv)} / {nsys} = {np.mean(conv):.3f}')
    print(f'mean error: {err.mean()}')
    print(f'mean absolute error: {np.abs(err).mean()}')
    if train_idx is not None:
        if test_idx is None:
            test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
        print(f'  training: {np.abs(err[train_idx]).mean()}')
        print(f'  testing: {np.abs(err[test_idx]).mean()}')
        print(f'mean absolute error after shift: {np.abs(err - err[train_idx].mean()).mean()}')
        print(f'  training: {np.abs(err[train_idx] - err[train_idx].mean()).mean()}')
        print(f'  testing: {np.abs(err[test_idx] - err[train_idx].mean()).mean()}')
    

def make_label(sys_dir, eref, fref=None):
    eref = eref.reshape(-1,1)
    nmol = eref.shape[0]
    ehf = np.load(f'{sys_dir}/e_hf.npy')
    assert ehf.shape[0] == nmol
    ecc = eref - ehf
    np.save(f'{sys_dir}/l_e_delta.npy', ecc)
    if fref is not None:
        fref = fref.reshape(nmol, -1, 3)
        fhf = np.load(f'{sys_dir}/f_hf.npy')
        assert fhf.shape == fref.shape
        fcc = fref - fhf
        np.save(f'{sys_dir}/l_f_delta.npy', fcc)


def collect_data(train_idx, test_idx=None, 
                 sys_dir="results", ene_ref="e_ref.npy", force_ref=None,
                 dump_dir=".", verbose=True):
    erefs = check_array(ene_ref).reshape(-1)
    nsys = erefs.shape[0]
    if nsys == 1 and "e_cf.npy" in os.listdir(sys_dir):
        systems = [os.path.abspath(sys_dir)]
    else:
        systems = sorted(map(os.path.abspath, glob.glob(f"{sys_dir}/*")))
    assert nsys == len(systems)

    convs = []
    ecfs = []
    for sys_i, ec_i in zip(systems, erefs):
        e0_i = np.load(os.path.join(sys_i, "e_hf.npy"))
        ecc_i = ec_i - e0_i
        np.save(os.path.join(sys_i, "l_e_delta.npy"), ecc_i)
        convs.append(np.load(os.path.join(sys_i, "conv.npy")))
        ecfs.append(np.load(os.path.join(sys_i, "e_cf.npy")))
    convs = np.array(convs).reshape(-1)
    ecfs = np.array(ecfs).reshape(-1)
    err = erefs - ecfs

    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
    if verbose:
        print(sys_dir)
        print_stat_per_sys(err, convs, train_idx, test_idx)
    
    np.savetxt(f'{dump_dir}/train_paths.raw', np.array(systems)[train_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/test_paths.raw', np.array(systems)[test_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/e_result.out', np.stack([erefs, ecfs], axis=-1), header="real pred")
    
    
def collect_data_grouped(train_idx, test_idx=None, 
                         sys_dir="results", ene_ref="e_ref.npy", force_ref=None,
                         dump_dir=".", append=True, verbose=True):
    eref = check_array(ene_ref).reshape(-1, 1)
    fref = check_array(force_ref)
    nmol = eref.shape[0]
    if not os.path.exists(f'{sys_dir}/e_cf.npy'):
        concat_data(sys_dir, dump_dir=sys_dir)    
    ecf = np.load(f'{sys_dir}/e_cf.npy').reshape(-1, 1)
    assert ecf.shape[0] == nmol, f"{ene_ref} ref size: {nmol}, {sys_dir} data size: {ecf.shape[0]}"
    make_label(sys_dir, eref, fref)
    # ehf = np.load(f'{sys_dir}/e_hf.npy')
    # np.save(f'{sys_dir}/l_e_delta.npy', eref - ehf)

    err = eref - ecf
    conv = np.load(f'{sys_dir}/conv.npy').reshape(-1)
    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nmol), train_idx, assume_unique=True)
    if verbose:
        print(sys_dir)
        print_stat_per_sys(err.reshape(-1), conv.reshape(-1), train_idx, test_idx)
    
    dd = [name for name in os.listdir(sys_dir) if ".npy" in name]
    os.makedirs(f'{sys_dir}/train', exist_ok=True)
    os.makedirs(f'{sys_dir}/test', exist_ok=True)
    for d in dd:
        np.save(f"{sys_dir}/train/{d}", np.load(f'{sys_dir}/{d}')[train_idx])
    for d in dd:
        np.save(f"{sys_dir}/test/{d}", np.load(f'{sys_dir}/{d}')[test_idx])
    shutil.copy(f'{sys_dir}/system.raw', f'{sys_dir}/train')
    shutil.copy(f'{sys_dir}/system.raw', f'{sys_dir}/test')
    # np.savetxt(f'{dump_dir}/train_paths.raw', [os.path.abspath(f'{dump_dir}/train')], fmt='%s')
    # np.savetxt(f'{dump_dir}/test_paths.raw', [os.path.abspath(f'{dump_dir}/test')], fmt='%s')
    # Path(f'{dump_dir}/train_paths.raw').write_text(str(Path(f'{dump_dir}/train').absolute()))
    # Path(f'{dump_dir}/test_paths.raw').write_text(str(Path(f'{dump_dir}/test').absolute()))
    mode = "a" if append else "w"
    with open(f'{dump_dir}/train_paths.raw', mode) as fp:
        fp.write(os.path.abspath(f'{sys_dir}/train') + "\n")
    with open(f'{dump_dir}/test_paths.raw', mode) as fp:
        fp.write(os.path.abspath(f'{sys_dir}/test') + "\n")


if __name__ == "__main__":
    cli()