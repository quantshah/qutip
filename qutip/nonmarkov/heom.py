# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
#                      Neill Lambert, Anubhav Vardhan, Alexander Pitchford.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module provides exact solvers for a system-bath setup using the
hierarchy equations of motion (HEOM).
"""

# Authors: Neill Lambert, Anubhav Vardhan, Alexander Pitchford
# Contact: nwlambert@gmail.com

import timeit
import numpy as np
#from scipy.misc import factorial
import scipy.sparse as sp
import scipy.integrate
from scipy.integrate import quad

from copy import copy
from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from qutip import liouvillian, mat2vec, state_number_enumerate
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.cy.heom import cy_pad_csr
from qutip.cy.spmath import zcsr_kron
from qutip.fastsparse import fast_csr_matrix, fast_identity
from functools import reduce
from operator import mul
from scipy.misc import factorial
from copy import copy


class HEOMSolver(object):
    """
    This is superclass for all solvers that use the HEOM method for
    calculating the dynamics evolution. There are many references for this.
    A good introduction, and perhaps closest to the notation used here is:
    DOI:10.1103/PhysRevLett.104.250401
    A more canonical reference, with full derivation is:
    DOI: 10.1103/PhysRevA.41.6676
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximation (RWA) for systems where the bath
    correlations can be approximated to a sum of complex eponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers (e.g. mesolve)

    This class should be treated as abstract. Currently the only subclass
    implemented is that for the Drude-Lorentz spectral density. This covers
    the majority of the work that has been done using this model, and there
    are some performance advantages to assuming this model where it is
    appropriate.

    There are opportunities to develop a more general spectral density code.

    Attributes
    ----------
    H_sys : Qobj
        System Hamiltonian

    coup_op : Qobj
        Operator describing the coupling between system and bath.

    coup_strength : float
        Coupling strength.

    temperature : float
        Bath temperature, in units corresponding to planck

    N_cut : int
        Cutoff parameter for the bath

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions

    planck : float
        reduced Planck constant

    boltzmann : float
        Boltzmann's constant

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    stats : :class:`qutip.solver.Stats`
        optional container for holding performance statitics
        If None is set, then statistics are not collected
        There may be an overhead in collecting statistics

    exp_coeff : list of complex
        Coefficients for the exponential series terms

    exp_freq : list of complex
        Frequencies for the exponential series terms
    """
    def __init__(self):
        raise NotImplementedError("This is a abstract class only. "
                "Use a subclass, for example HSolverDL")

    def reset(self):
        """
        Reset any attributes to default values
        """
        self.planck = 1.0
        self.boltzmann = 1.0
        self.H_sys = None
        self.coup_op = None
        self.coup_strength = 0.0
        self.temperature = 1.0
        self.N_cut = 10
        self.N_exp = 2
        self.N_he = 0

        self.exp_coeff = None
        self.exp_freq = None

        self.options = None
        self.progress_bar = None
        self.stats = None

        self.ode = None
        self.configured = False

    def configure(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used

        progress_bar: BaseProgressBar
            Optional instance of BaseProgressBar, or a subclass thereof, for
            showing the progress of the simulation.
            If set to None, then the default progress bar will be used
            Set to False for no progress bar

        stats: :class:`qutip.solver.Stats`
            Optional instance of solver.Stats, or a subclass thereof, for
            storing performance statistics for the solver
            If set to True, then the default Stats for this class will be used
            Set to False for no stats
        """

        self.H_sys = H_sys
        self.coup_op = coup_op
        self.coup_strength = coup_strength
        self.temperature = temperature
        self.N_cut = N_cut
        self.N_exp = N_exp
        if planck: self.planck = planck
        if boltzmann: self.boltzmann = boltzmann
        if isinstance(options, Options): self.options = options
        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None
        if isinstance(stats, Stats):
            self.stats = stats
        elif stats == True:
            self.stats = self.create_new_stats()
        elif stats == False:
            self.stats = None

    def create_new_stats(self):
        """
        Creates a new stats object suitable for use with this solver
        Note: this solver expects the stats object to have sections
            config
            integrate
        """
        stats = Stats(['config', 'run'])
        stats.header = "Hierarchy Solver Stats"
        return stats

class HSolverDL(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as an infinite sum of exponentials which depend on the
    temperature, these are called the Matsubara terms or Matsubara frequencies

    For practical computation purposes an approximation must be used based
    on a small number of Matsubara terms (typically < 4).

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann

    bnd_cut_approx : bool
        Use boundary cut off approximation
        Can be
    """

    def __init__(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=1.0, boltzmann=1.0,
                     renorm=True, bnd_cut_approx=True,
                     options=None, progress_bar=None, stats=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.configure(H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=planck, boltzmann=boltzmann,
                     renorm=renorm, bnd_cut_approx=bnd_cut_approx, stats=stats)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Calls configure from :class:`HEOMSolver` and sets any attributes
        that are specific to this subclass
        """
        start_config = timeit.default_timer()

        HEOMSolver.configure(self, H_sys, coup_op, coup_strength,
                    temperature, N_cut, N_exp,
                    planck=planck, boltzmann=boltzmann,
                    options=options, progress_bar=progress_bar, stats=stats)
        self.cut_freq = cut_freq
        if renorm is not None: self.renorm = renorm
        if bnd_cut_approx is not None: self.bnd_cut_approx = bnd_cut_approx

        # Load local values for optional parameters
        # Constants and Hamiltonian.
        hbar = self.planck
        options = self.options
        progress_bar = self.progress_bar
        stats = self.stats


        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                ss_conf = stats.add_section('config')

        c, nu = self._calc_matsubara_params()

        if renorm:
            norm_plus, norm_minus = self._calc_renorm_factors()
            if stats:
                stats.add_message('options', 'renormalisation', ss_conf)
        # Dimensions et by system
        N_temp = 1
        for i in H_sys.dims[0]:
            N_temp *= i
        sup_dim = N_temp**2
        unit_sys = qeye(N_temp)


        # Use shorthands (mainly as in referenced PRL)
        lam0 = self.coup_strength
        gam = self.cut_freq
        N_c = self.N_cut
        N_m = self.N_exp
        Q = coup_op # Q as shorthand for coupling operator
        beta = 1.0/(self.boltzmann*self.temperature)

        # Ntot is the total number of ancillary elements in the hierarchy
        # Ntot = factorial(N_c + N_m) / (factorial(N_c)*factorial(N_m))
        # Turns out to be the same as nstates from state_number_enumerate
        N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*N_m , N_c)

        unit_helems = fast_identity(N_he)
        if self.bnd_cut_approx:
            # the Tanimura boundary cut off operator
            if stats:
                stats.add_message('options', 'boundary cutoff approx', ss_conf)
            op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

            approx_factr = ((2*lam0 / (beta*gam*hbar)) - 1j*lam0) / hbar
            for k in range(N_m):
                approx_factr -= (c[k] / nu[k])
            L_bnd = -approx_factr*op.data
            L_helems = zcsr_kron(unit_helems, L_bnd)
        else:
            L_helems = fast_csr_matrix(shape=(N_he*sup_dim, N_he*sup_dim))

        # Build the hierarchy element interaction matrix
        if stats: start_helem_constr = timeit.default_timer()

        unit_sup = spre(unit_sys).data
        spreQ = spre(Q).data
        spostQ = spost(Q).data
        commQ = (spre(Q) - spost(Q)).data
        N_he_interact = 0

        for he_idx in range(N_he):
            he_state = list(idx2he[he_idx])
            n_excite = sum(he_state)

            # The diagonal elements for the hierarchy operator
            # coeff for diagonal elements
            sum_n_m_freq = 0.0
            for k in range(N_m):
                sum_n_m_freq += he_state[k]*nu[k]

            op = -sum_n_m_freq*unit_sup
            L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx)
            L_helems += L_he

            # Add the neighour interations
            he_state_neigh = copy(he_state)
            for k in range(N_m):

                n_k = he_state[k]
                if n_k >= 1:
                    # find the hierarchy element index of the neighbour before
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k - 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = c[k]*spreQ - np.conj(c[k])*spostQ
                    if renorm:
                        op = -1j*norm_minus[n_k, k]*op
                    else:
                        op = -1j*n_k*op

                    L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

                if n_excite <= N_c - 1:
                    # find the hierarchy element index of the neighbour after
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k + 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = commQ
                    if renorm:
                        op = -1j*norm_plus[n_k, k]*op
                    else:
                        op = -1j*op

                    L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

        if stats:
            stats.add_timing('hierarchy contruct',
                             timeit.default_timer() - start_helem_constr,
                            ss_conf)
            stats.add_count('Num hierarchy elements', N_he, ss_conf)
            stats.add_count('Num he interactions', N_he_interact, ss_conf)

        # Setup Liouvillian
        if stats: 
            start_louvillian = timeit.default_timer()
        
        H_he = zcsr_kron(unit_helems, liouvillian(H_sys).data)

        L_helems += H_he

        if stats:
            stats.add_timing('Liouvillian contruct',
                             timeit.default_timer() - start_louvillian,
                            ss_conf)

        if stats: start_integ_conf = timeit.default_timer()

        r = scipy.integrate.ode(cy_ode_rhs)

        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('Liouvillian contruct',
                             time_now - start_integ_conf,
                            ss_conf)
            if ss_conf.total_time is None:
                ss_conf.total_time = time_now - start_config
            else:
                ss_conf.total_time += time_now - start_config

        self._ode = r
        self._N_he = N_he
        self._sup_dim = sup_dim
        self._configured = True

    def run(self, rho0, tlist):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system.

        tlist : list
            Time over which system evolves.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
        """

        start_run = timeit.default_timer()

        sup_dim = self._sup_dim
        stats = self.stats
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")
        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                raise RuntimeError("No config section for solver stats")
            ss_run = stats.sections.get('run')
            if ss_run is None:
                ss_run = stats.add_section('run')

        # Set up terms of the matsubara and tanimura boundaries
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies

        Returns
        -------
        c, nu: both list(float)

        """
        c = []
        nu = []
        lam0 = self.coup_strength
        gam = self.cut_freq
        hbar = self.planck
        beta = 1.0/(self.boltzmann*self.temperature)
        N_m = self.N_exp

        g = 2*np.pi / (beta*hbar)
        for k in range(N_m):
            if k == 0:
                nu.append(gam)
                c.append(lam0*gam*
                    (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
            else:
                nu.append(k*g)
                c.append(4*lam0*gam*nu[k] /
                      ((nu[k]**2 - gam**2)*beta*hbar**2))

        self.exp_coeff = c
        self.exp_freq = nu
        return c, nu

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m] of float
        """
        c = self.exp_coeff
        N_m = self.N_exp
        N_c = self.N_cut

        norm_plus = np.empty((N_c+1, N_m))
        norm_minus = np.empty((N_c+1, N_m))
        for k in range(N_m):
            for n in range(N_c+1):
                norm_plus[n, k] = np.sqrt(abs(c[k])*(n + 1))
                norm_minus[n, k] = np.sqrt(float(n)/abs(c[k]))

        return norm_plus, norm_minus


def _pad_csr(A, row_scale, col_scale, insertrow=0, insertcol=0):
    """
    Expand the input csr_matrix to a greater space as given by the scale.
    Effectively inserting A into a larger matrix
         zeros([A.shape[0]*row_scale, A.shape[1]*col_scale]
    at the position [A.shape[0]*insertrow, A.shape[1]*insertcol]
    The same could be achieved through using a kron with a matrix with
    one element set to 1. However, this is more efficient
    """

    # ajgpitch 2016-03-08:
    # Clearly this is a very simple operation in dense matrices
    # It seems strange that there is nothing equivalent in sparse however,
    # after much searching most threads suggest directly addressing
    # the underlying arrays, as done here.
    # This certainly proved more efficient than other methods such as stacking
    #TODO: Perhaps cythonize and move to spmatfuncs

    if not isinstance(A, sp.csr_matrix):
        raise TypeError("First parameter must be a csr matrix")
    nrowin = A.shape[0]
    ncolin = A.shape[1]
    nrowout = nrowin*row_scale
    ncolout = ncolin*col_scale

    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        A.indices = A.indices + insertcol*ncolin
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        A.indptr = np.concatenate((A.indptr,
                        np.array([A.indptr[-1]]*(row_scale-1)*nrowin)))
    elif insertrow == row_scale-1:
        A.indptr = np.concatenate((np.array([0]*(row_scale - 1)*nrowin),
                                   A.indptr))
    elif insertrow > 0 and insertrow < row_scale - 1:
         A.indptr = np.concatenate((np.array([0]*insertrow*nrowin), A.indptr,
                np.array([A.indptr[-1]]*(row_scale - insertrow - 1)*nrowin)))
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return A


def _heom_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.
    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.
    excitations : integer
        The maximum numbers of dimension
    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1
    return nstates, state2idx, idx2state


def _heom_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.
    Example:
        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.
    state : list
        Current state in the iteration. Used internally.
    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.
    idx : integer
        Current index in the iteration. Used internally.
    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.
    """

    if state is None:
        state = np.zeros(len(dims))

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
            
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


class HSolverUnderdampedBrownian(HEOMSolver):
    """
    HEOM solver based on the underdamped Brownian motion spectral density.

    :math:`J(\omega) = \frac{\gamma \lambda^2 \omega}{(\omega^2 - \omega_0^2)^2 + \gamma^2 \omega}`

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann.

    bnd_cut_approx : bool
        Use boundary cut off approximation.
    """
    def __init__(self, H_sys, coup_op, coup_strength, ckA, vkA,
                 ck_corr, vk_corr,
                 temperature, N_cut, N_exp, cut_freq, planck=1.0,
                 boltzmann=1.0, renorm=True, bnd_cut_approx=True,
                 options=None, progress_bar=None, stats=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.liouvillian = None
        self.H_sys = H_sys
        self.coup_op = coup_op
        self.coup_strength = coup_strength
        self.temperature = temperature
        self.N_cut = N_cut
        self.N_exp = N_exp
        self.cut_freq = cut_freq
        self.configure(ckA, vkA, ck_corr, vk_corr)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, ckA, vkA, ck_corr, vk_corr):
        """
        Configures the HEOM hierarchy.

        Parameters
        ----------
        ckA: list
            The list of coefficients for the non-Matsubara part of the
            spectral density.

        vkA: list
            The list of frequencies for the non-Matsubara part of the
            expansion of the spectral density.

        ck_corr: list
            The list of coefficients for the Matsubara part of the
            spectral density.

        vk_corr: list
            The list of frequencies for the Matsubara part of the
            expansion of the spectral density.
        """
        H = self.H_sys
        Q = self.coup_op
        lam = self.coup_strength
        Nc = self.N_cut
        N = self.N_exp
        #Parameters and hamiltonian

        hbar = self.planck
        kb = self.boltzmann

        N_temp = reduce(mul, H.dims[0], 1)
        Nsup = N_temp**2
        unit = qeye(N_temp)

        #Ntot is the total number of ancillary elements in the hierarchy
        Ntot = int(round(factorial(Nc+N) / (factorial(Nc) * factorial(N))))
        LD1 = -2.* spre(Q) * spost(Q.dag()) + spre(Q.dag() * Q) + spost(Q.dag() * Q)

        c0=ckA[0]
        pref=0.
        L12=0.*LD1;
        
        #Setup liouvillian

        L = liouvillian(H, [L12])
        Ltot = L.data
        unitthing=sp.identity(Ntot, dtype='complex', format='csr')
        Lbig = sp.kron(unitthing,Ltot.tocsr())
        
        nstates, state2idx, idx2state =_heom_state_dictionaries([Nc+1]*(N),Nc)
        for nlabelt in _heom_number_enumerate([Nc+1]*(N),Nc):
            nlabel = list(nlabelt)                    
            ntotalcheck = 0
            for ncheck in range(N):
                ntotalcheck = ntotalcheck + nlabel[ncheck]                            
            current_pos = int(round(state2idx[tuple(nlabel)]))
            Ltemp = sp.lil_matrix((Ntot, Ntot))
            Ltemp[current_pos,current_pos] = 1.
            Ltemp.tocsr()
            Lbig = Lbig + sp.kron(Ltemp,(-nlabel[0] * vkA[0] * spre(unit).data))
            Lbig = Lbig + sp.kron(Ltemp,(-nlabel[1] * vkA[1] * spre(unit).data))
            #bi-exponential corrections:
            if N==3:
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[2] * vk_corr[0] * spre(unit).data))
            if N==4:
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[2] * vk_corr[0] * spre(unit).data))
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[3] * vk_corr[1] * spre(unit).data))
            
            for kcount in range(N):
                if nlabel[kcount]>=1:
                #find the position of the neighbour
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] -1
                    current_pos2 = int(round(state2idx[tuple(nlabel)]))
                    Ltemp = sp.lil_matrix(np.zeros((Ntot,Ntot)))
                    Ltemp[current_pos, current_pos2] = 1
                    Ltemp.tocsr()
                # renormalized version:    
                    #ci =  (4 * lam0 * gam * kb * Temperature * kcount
                    #      * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                    if kcount==0:
                        
                        c0n=lam
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(c0n)))
                                         * (0.0*spre(Q).data
                                         - (lam)
                                         * spost(Q).data)))
                    if kcount==1:     
                        cin=lam
                        ci =  ckA[kcount]
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(cin)))
                                         * ((lam) * spre(Q).data
                                         - (0.0)
                                         * spost(Q).data)))
                        
                    if kcount==2:     
                        cin=ck_corr[0]                        
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                             * np.sqrt((nlabeltemp[kcount]
                                                / abs(cin)))
                                             * cin*(spre(Q).data - spost(Q).data)))
                    if kcount==3:     
                        cin=ck_corr[1]                        
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                             * np.sqrt((nlabeltemp[kcount]
                                                / abs(cin)))
                                             * cin*(spre(Q).data - spost(Q).data)))
                    nlabel = copy(nlabeltemp)

            for kcount in range(N):
                if ntotalcheck<=(Nc-1):
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] + 1
                    current_pos3 = int(round(state2idx[tuple(nlabel)]))
                if current_pos3<=(Ntot):
                    Ltemp = sp.lil_matrix(np.zeros((Ntot,Ntot)))
                    Ltemp[current_pos, current_pos3] = 1
                    Ltemp.tocsr()
                #renormalized   
                    if kcount==0:
                        c0n=lam
                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*((abs(c0n))))
                                      * (spre(Q)- spost(Q)).data)
                    if kcount==1:
                        ci =ckA[kcount]
                        cin=lam
                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                      * (spre(Q)- spost(Q)).data)
                    if kcount==2:
                        cin=ck_corr[0]
                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                      * (spre(Q)- spost(Q)).data)
                    if kcount==3:
                        cin=ck_corr[1]
                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                      * (spre(Q)- spost(Q)).data)    
                 
                nlabel = copy(nlabeltemp)
        self.liouvillian = Lbig
        return Lbig

    def run(self, rho0, tlist):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system.

        tlist : list
            Time over which system evolves.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
        """
        options = self.options
        stats = self.stats
        Nc = self.N_cut
        Nk = self.N_exp
        self._N_he = int(factorial(Nc + Nk)/(factorial(Nc)*factorial(Nk)))
        start_run = timeit.default_timer()
        r = scipy.integrate.ode(cy_ode_rhs)
        N_temp = 1
        for i in self.H_sys.dims[0]:
            N_temp *= i
        sup_dim = N_temp**2

        L_helems = self.liouvillian
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        # Set up terms of the matsubara and tanimura boundaries
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output


def underdamped_brownian(w, lam, gamma, w0):
    """
    Calculates the underdamped Brownian motion spectral density.

    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.

    lam: float
        The coupling strength parameter.

    gamma: float
        A parameter characterizing the FWHM of the spectral density.

    w0: float
        The qubit frequency.

    Returns
    -------
    spectral_density: np.ndarray
        The spectral density for specified parameters.
    """
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    prefactor = (lam**2)*gamma
    spectral_density = prefactor*(w/((w-a)*(w+a)*(w-aa)*(w+aa)))
    return spectral_density


def bath_correlation(spectral_density, tlist,
                     params, beta, w_cutoff):
    """
    Calculates the bath correlation function (C) for a specific spectral
    density (J(w)) for an environment modelled as a bath of harmonic
    oscillators. If :math: `\beta` is the inverse temperature of the bath
    then the correlation is:

    :math:`C(t) = \frac{1}{\pi} \left[\int_{0}^{\infty} \coth
    (\beta \omega /2) \cos(\omega t) - i\sin(\omega t) \right]`

    where :math: `\beta = 1/kT` with T as the bath temperature and k as
    the Boltzmann's constant.

    Assumptions:
        1. The bath is in a thermal state at a given temperature.
        2. The initial state of the environment is Gaussian.
        3. Bath operators are in a product state with the system intially.

    The `spectral_density` function is a callable, for example the Ohmic
    spectral density given as: `ohmic_sd = lambda w, eta: eta*w`

    Parameters
    ==========
    spectral_density: callable
        A function of the form f(w, *params) which calculates the spectral
        densities for the given parameters, where w are the frequencies.

    tlist : *list* / *array*
        A 1D array/list of times to calculate the correlation.

    params: ndarray
        A 1D array of parameters for the spectral density function.

    w_cutoff: float
        The cutoff value for the angular frequencies
        for integration.

        In general the intergration is for all values but since at
        higher frequencies, the spectral density is zero, we set
        a finite limit to the numerical integration.

    beta: float
        The inverse temperature of the bath. If the temperature
        is zero, `beta` goes to infinity and we can replace the coth(x)
        term in the correlation function's real part with 1.
        At higher temperatures the coth(x) function behaves poorly at
        low frequencies.

    Returns
    =======
    corr: ndarray
        A 1D array giving the values of the correlation function for given
        time.
    """
    if not callable(spectral_density):
        raise TypeError("""Spectral density should be a callable function
            f(w, args)""")

    corrR = []
    corrI = []

    coth = lambda x: 1/np.tanh(x)
    w_start = 0.

    integrandR = lambda w, t: np.real(spectral_density(w, *params) \
        *(coth(beta*(w/2)))*np.cos(w*t))
    integrandI = lambda w, t: np.real(-spectral_density(w, *params) \
        *np.sin(w*t))

    for i in tlist:
        corrR.append(np.real(quad(integrandR, w_start, w_cutoff, args=(i,))[0]))
        corrI.append(quad(integrandI, w_start, w_cutoff, args=(i,))[0])
    corr = (np.array(corrR) + 1j*np.array(corrI))/np.pi
    return corr
