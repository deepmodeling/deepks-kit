import os
import sys
import glob
import numpy as np
import shutil
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.utils import check_list, check_array
from deepks.utils import load_array, load_yaml
from deepks.utils import get_sys_name, get_with_prefix


def concat_data(systems=None, sys_dir=".", dump_dir=".", pattern="*"):
    if systems is None:
        systems = sorted(filter(os.path.isdir, 
            map(os.path.abspath, glob.glob(f"{sys_dir}/{pattern}"))))
    npy_names = list(map(os.path.basename, glob.glob(f"{systems[0]}/*.npy")))
    os.makedirs(dump_dir, exist_ok=True)
    for nm in npy_names:
        tmp_array = np.concatenate([np.load(f"{sys}/{nm}") for sys in systems])
        np.save(f"{dump_dir}/{nm}", tmp_array)
    if os.path.exists(f'{systems[0]}/system.raw'):
        shutil.copy(f'{systems[0]}/system.raw', dump_dir)


def print_stats(systems=None, test_sys=None, 
               dump_dir=None, test_dump=None, group=False,
               with_conv=True, with_e=True, e_name="e_tot", 
               with_f=True, f_name="f_tot",
               with_s=True, s_name="s_tot",
               with_o=True, o_name="o_tot"):
    load_func = load_stat if not group else load_stat_grouped
    if dump_dir is None:
        dump_dir = "."
    if test_dump is None:
        test_dump = dump_dir
    shift = None
    if systems is not None:
        tr_c, tr_e, tr_f, tr_s, tr_o = load_func(systems, dump_dir, with_conv, 
                                     with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name)
        print("Training:")
        if tr_c is not None:
            print_stats_conv(tr_c, indent=2)
        if tr_e is not None:
            shift = tr_e.mean()
            print_stats_e(tr_e, shift=shift, indent=2)
        if tr_f is not None:
            print_stats_f(tr_f, indent=2)
        if tr_s is not None:
            print_stats_s(tr_s, indent=2)
        if tr_o is not None:
            print_stats_o(tr_o, indent=2)
    if test_sys is not None:
        ts_c, ts_e, ts_f, ts_s, ts_o = load_func(test_sys, test_dump, with_conv, 
                                     with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name)
        print("Testing:")
        if ts_c is not None:
            print_stats_conv(ts_c, indent=2)
        if ts_e is not None:
            print_stats_e(ts_e, shift=shift, indent=2)
        if ts_f is not None:
            print_stats_f(ts_f, indent=2)
        if ts_s is not None:
            print_stats_s(ts_s, indent=2)
        if ts_o is not None:
            print_stats_o(ts_o, indent=2)

def print_stats_conv(conv, indent=0):
    nsys = conv.shape[0]
    ind = " "*indent
    print(ind+f'Convergence:')
    print(ind+f'  {np.sum(conv)} / {nsys} = \t {np.mean(conv):.5f}')


def print_stats_e(e_err, shift=None, indent=0):
    ind = " "*indent
    print(ind+"Energy:")
    print(ind+f'  ME: \t {e_err.mean()}')
    print(ind+f'  MAE: \t {np.abs(e_err).mean()}')
    if shift:
        print(ind+f'  MARE: \t {np.abs(e_err-shift).mean()}')


def print_stats_f(f_err, indent=0):
    ind = " "*indent
    print(ind+"Force:")
    print(ind+f'  MAE: \t {np.abs(f_err).mean()}')

def print_stats_s(s_err, indent=0):
    ind = " "*indent
    print(ind+"Stress:")
    print(ind+f'  MAE: \t {np.abs(s_err).mean()}')

def print_stats_o(o_err, indent=0):
    ind = " "*indent
    print(ind+"Band gap:")
    print(ind+f'  ME: \t {o_err.mean()}')
    print(ind+f'  MAE: \t {np.abs(o_err).mean()}')

def load_stat(systems, dump_dir,
              with_conv=True, with_e=True, e_name="e_tot", 
              with_f=True, f_name="f_tot",
              with_s=True, s_name="s_tot",
              with_o=True, o_name="o_tot"):
    systems = check_list(systems)
    c_res = []
    e_err = []
    f_err = []
    s_err = []
    o_err = []
    for fl in systems:
        lbase = get_sys_name(fl)
        rbase = os.path.join(dump_dir, os.path.basename(lbase))
        if with_conv:
            try:
                c_res.append(load_array(get_with_prefix("conv", rbase, ".npy")))
            except FileNotFoundError as e:
                print("Warning! conv.npy not found:", e, file=sys.stderr)
        if with_e:
            try:
                re = load_array(get_with_prefix(e_name, rbase, ".npy")).reshape(-1,1)
                le = load_array(get_with_prefix("energy", lbase, ".npy")).reshape(-1,1)
                e_err.append(le - re)
            except FileNotFoundError as e:
                print("Warning! energy file not found:", e, file=sys.stderr)
        if with_f:
            try:
                rf = load_array(get_with_prefix(f_name, rbase, ".npy"))
                lf = load_array(get_with_prefix("force", lbase, ".npy")).reshape(rf.shape)
                f_err.append(np.abs(lf - rf).mean((-1,-2)))
            except FileNotFoundError as e:
                print("Warning! force file not found:", e, file=sys.stderr)
        if with_s:
            try:
                rs = load_array(get_with_prefix(s_name, rbase, ".npy"))
                ls = load_array(get_with_prefix("stress", lbase, ".npy"))[:,[0,1,2,4,5,8]] #extract the upper-triangle part
                ls = ls.reshape(rs.shape)
                s_err.append(np.abs(ls - rs))
            except FileNotFoundError as e:
                print("Warning! stress file not found:", e, file=sys.stderr)
        if with_o:
            try:
                ro = load_array(get_with_prefix(o_name, rbase, ".npy"))
                lo = load_array(get_with_prefix("orbital", lbase, ".npy")).reshape(ro.shape)
                o_err.append(np.abs(lo - ro))
            except FileNotFoundError as e:
                print("Warning! orbital file not found:", e, file=sys.stderr)
    return np.concatenate(c_res, 0) if c_res else None, \
           np.concatenate(e_err, 0) if e_err else None, \
           np.concatenate(f_err, 0) if f_err else None, \
           np.concatenate(s_err, 0) if s_err else None, \
           np.concatenate(o_err, 0) if o_err else None


def load_stat_grouped(systems, dump_dir=".",
                      with_conv=True, with_e=True, e_name="e_tot", 
                      with_f=True, f_name="f_tot",
                      with_s=True, s_name="s_tot",
                      with_o=True, o_name="o_tot"):
    systems = check_list(systems)
    lbases = [get_sys_name(fl) for fl in systems]
    c_res = e_err = f_err = s_err = o_err = None
    if with_conv:
        c_res = load_array(get_with_prefix("conv", dump_dir, ".npy"))
    if with_e:
        e_res = load_array(get_with_prefix(e_name, dump_dir, ".npy"))
        e_lbl = np.concatenate([
            load_array(get_with_prefix("energy", lb, ".npy")) for lb in lbases
        ], 0).reshape(-1,1)
        e_err = e_lbl - e_res
    if with_f:
        f_res = load_array(get_with_prefix(f_name, dump_dir, ".npy"))
        f_lbl = np.concatenate([
            load_array(get_with_prefix("force", lb, ".npy")) for lb in lbases
        ], 0).reshape(f_res.shape)
        f_err = f_lbl - f_res
    if with_s:
        s_res = load_array(get_with_prefix(s_name, dump_dir, ".npy"))
        s_lbl = np.concatenate([
            load_array(get_with_prefix("stress", lb, ".npy"))[:,[0,1,2,4,5,8]] for lb in lbases
        ], 0).reshape(s_res.shape) #extract the upper-triangle part
        s_err = s_lbl - s_res
    if with_o:
        o_res = load_array(get_with_prefix(o_name, dump_dir, ".npy"))
        o_lbl = np.concatenate([
            load_array(get_with_prefix("orbital", lb, ".npy")) for lb in lbases
        ], 0).reshape(o_res.shape)
        o_err = o_lbl - o_res
    return c_res, e_err, f_err, s_err, o_err


# Below are legacy tools, kept for old examples

def print_stats_per_sys(err, conv=None, train_idx=None, test_idx=None):
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
    ehf = np.load(f'{sys_dir}/e_base.npy')
    assert ehf.shape[0] == nmol
    ecc = eref - ehf
    np.save(f'{sys_dir}/l_e_delta.npy', ecc)
    if fref is not None:
        fref = fref.reshape(nmol, -1, 3)
        fhf = np.load(f'{sys_dir}/f_base.npy')
        assert fhf.shape == fref.shape
        fcc = fref - fhf
        np.save(f'{sys_dir}/l_f_delta.npy', fcc)


def collect_data(train_idx, test_idx=None, 
                 sys_dir="results", ene_ref="e_ref.npy", force_ref=None,
                 dump_dir=".", verbose=True):
    erefs = check_array(ene_ref).reshape(-1)
    nsys = erefs.shape[0]
    if nsys == 1 and "e_tot.npy" in os.listdir(sys_dir):
        systems = [os.path.abspath(sys_dir)]
    else:
        systems = sorted(map(os.path.abspath, glob.glob(f"{sys_dir}/*")))
    assert nsys == len(systems)

    convs = []
    ecfs = []
    for sys_i, ec_i in zip(systems, erefs):
        e0_i = np.load(os.path.join(sys_i, "e_base.npy"))
        ecc_i = ec_i - e0_i
        np.save(os.path.join(sys_i, "l_e_delta.npy"), ecc_i)
        convs.append(np.load(os.path.join(sys_i, "conv.npy")))
        ecfs.append(np.load(os.path.join(sys_i, "e_tot.npy")))
    convs = np.array(convs).reshape(-1)
    ecfs = np.array(ecfs).reshape(-1)
    err = erefs - ecfs

    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
    if verbose:
        print(sys_dir)
        print_stats_per_sys(err, convs, train_idx, test_idx)
    
    np.savetxt(f'{dump_dir}/train_paths.raw', np.array(systems)[train_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/test_paths.raw', np.array(systems)[test_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/e_result.out', np.stack([erefs, ecfs], axis=-1), header="real pred")
    
    
def collect_data_grouped(train_idx, test_idx=None, 
                         sys_dir="results", ene_ref="e_ref.npy", force_ref=None,
                         dump_dir=".", append=True, verbose=True):
    eref = check_array(ene_ref).reshape(-1, 1)
    fref = check_array(force_ref)
    nmol = eref.shape[0]
    if not os.path.exists(f'{sys_dir}/e_tot.npy'):
        concat_data(sys_dir, dump_dir=sys_dir)    
    ecf = np.load(f'{sys_dir}/e_tot.npy').reshape(-1, 1)
    assert ecf.shape[0] == nmol, f"{ene_ref} ref size: {nmol}, {sys_dir} data size: {ecf.shape[0]}"
    make_label(sys_dir, eref, fref)
    # ehf = np.load(f'{sys_dir}/e_base.npy')
    # np.save(f'{sys_dir}/l_e_delta.npy', eref - ehf)

    err = eref - ecf
    conv = np.load(f'{sys_dir}/conv.npy').reshape(-1)
    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nmol), train_idx, assume_unique=True)
    if verbose:
        print(sys_dir)
        print_stats_per_sys(err.reshape(-1), conv.reshape(-1), train_idx, test_idx)
    
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
    from deepks.main import stats_cli as cli
    cli()