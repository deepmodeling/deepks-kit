import time
import numpy as np
from pyscf.dft import numint, gen_grid
from pyscf.lib import logger

class AbstructPenalty(object):
    """
    Abstruct class for penalty term in scf hamiltonian.
    To implement a penalty one needs to implement 
    fock_hook and (optional) init_hook methods.
    """

    def init_hook(self, mf, **envs):
        """
        Method to be called when initialize the scf object.
        Used to initialize the penalty with molecule info.
        """
        pass

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        """
        Method to be called before get_fock is called.
        The returned matrix would be added to the vhf matrix
        """
        raise NotImplementedError("fock_hook method is not implemented")


class DummyPenalty(AbstructPenalty):
    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        return 0


class DensityPenalty(AbstructPenalty):
    r"""
    penalty on the difference w.r.t target density
    E_p = \lambda / 2 * \int dx (\rho(x) - \rho_target(x))^2
    V_p = \lambda * \int dx <ao_i|x> (\rho(x) - \rho_target(x)) <x|ao_j> 
    The target density should be given as density matrix in ao basis
    """

    def __init__(self, target_dm, strength=1, random=False, start_cycle=0):
        if isinstance(target_dm, str):
            target_dm = np.load(target_dm)
        self.dm_t = target_dm
        self.init_strength = strength
        self.strength = strength * np.random.rand() if random else strength
        self.start_cycle = start_cycle
        # below are values to be initialized later in init_hook
        self.grids = None
        self.ao_value = None

    def init_hook(self, mf, **envs):
        if hasattr(mf, "grid"):
            self.grids = mf.grids
        else:
            self.grids = gen_grid.Grids(mf.mol)

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        # cycle > 0 means it is doing scf iteration
        if 0 <= cycle < self.start_cycle:
            return 0
        if self.grids.coords is None:
            self.grids.build()
        if self.ao_value is None:
            self.ao_value = numint.eval_ao(mf.mol, self.grids.coords, deriv=0)
        tic = (time.clock(), time.time())
        rho_diff = numint.eval_rho(mf.mol, self.ao_value, dm - self.dm_t)
        v_p = numint.eval_mat(mf.mol, self.ao_value, self.grids.weights, rho_diff, rho_diff)
        # cycle < 0 means it is just checking, we only print here
        if cycle < 0 and mf.verbose >=4:
            diff_norm = np.sum(np.abs(rho_diff)*self.grids.weights)
            logger.info(mf, f"  Density Penalty: |diff| = {diff_norm}")
            logger.timer(mf, "dens_pnt", *tic)
        return self.strength * v_p


class CoulombPenalty(AbstructPenalty):
    r"""
    penalty given by the coulomb energy of density difference

    """

    def __init__(self, target_dm, strength=1, random=False, start_cycle=0):
        if isinstance(target_dm, str):
            target_dm = np.load(target_dm)
        self.dm_t = target_dm
        self.init_strength = strength
        self.strength = strength * np.random.rand() if random else strength
        self.start_cycle = start_cycle
        # below are values to be initialized later in init_hook
        self.vj_t = None

    def init_hook(self, mf, **envs):
        self.vj_t = mf.get_j(dm=self.dm_t)

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        # cycle > 0 means it is doing scf iteration
        if 0 <= cycle < self.start_cycle:
            return 0
        tic = (time.clock(), time.time())
        vj = mf.get_j(dm=dm)
        v_p = vj - self.vj_t
        # cycle < 0 means it is just checking, we only print here
        if cycle < 0 and mf.verbose >=4:
            diff_norm = np.einsum("ij,ij", dm, v_p)
            logger.info(mf, f"  Coulomb Penalty: |diff| = {diff_norm}")
            logger.timer(mf, "coul_pnt", *tic)
        return self.strength * v_p