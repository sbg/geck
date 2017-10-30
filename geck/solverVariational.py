# geck: Genotype Error Comparator Kit, for benchmarking genotyping tools
# Copyright (C) 2017 Seven Bridges Genomics Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.special import digamma, gammaln

from model import GeckModel


class GeckSolverVariational(GeckModel):
    """Variational solution described in Appendix B in Notes.pdf

    :ivar n_array: aggregate trio genotype counts (27x27-array)

    :ivar f_alpha: Dirichlet parameter of trio frequencies (f) (15-array)
    :ivar f_alpha_prior: parameter of prior distribution of f ((15-array)
    :ivar theta_alpha: Dirichlet parameter of fraction (theta)
                       of variants affected by different error models,
                       in the 3 groups (3x15-array)
    :ivar theta_alpha_prior: parameter of prior distribution of theta
                             (3x15-array)
    :ivar e_alpha: Dirichlet parameter of genotyping error rates (E)
                             in the 15 different modes (15x3x3x3-array)
    :ivar e_alpha_prior: parameter of prior distribution of E (15x3x3x3-array)

    :ivar ev_log_f: ExpValue(log(f)) (15-array)
    :ivar ev_log_theta: ExpValue(log(theta)) (3x15-array)
    :ivar ev_log_e: ExpValue(log(E) (15x3x3x3-array)
    :ivar r: P(g, m | G1, G2) (27x27x15x15-array)

    :ivar n_array_complete_samples: list of N_complete samples,
            the prediction of the model (list of 27x27x15x15-arrays)

    """
    def __init__(self):
        GeckModel.__init__(self)

        self.n_array = (-1) * np.ones((len(self.fg), len(self.fg)))

        self.f_alpha = (-1) * np.ones(len(self.vfg))
        self.f_alpha_prior = 1.0 * np.ones_like(self.f_alpha)
        self.theta_alpha = (-1) * np.ones((len(self.groups), len(self.modes)))
        self.theta_alpha_prior = 1.0 * self.theta_mask.copy()
        self.e_alpha = (-1) * np.ones((len(self.modes), len(self.ig),
                                       len(self.ig), len(self.ig)))
        self.e_alpha_prior = 1.0 * self.e_array_mask.copy()

        self.ev_log_f = (-1) * np.ones_like(self.f_alpha)
        self.ev_log_theta = (-1) * np.ones_like(self.theta_alpha)
        self.ev_log_e = (-1) * np.ones_like(self.e_alpha)
        self.r = (-1) * np.ones((len(self.fg), len(self.fg),
                                 len(self.vfg), len(self.modes)))

        self.n_array_complete_samples = []

    def import_data(self, geck_data):
        """Gets aggregate data from a GeckData instance

        Gets aggregate data from a GeckData instance,
        while making sure the ordering matches with self.fg

        :param geck_data: input data (joint histogram, N)
        :type geck_data: GeckData
        :return: None (sets self.n_array)

        """
        self.n_array = geck_data.get_data_matrix(self.fg)

    def init_f_alpha(self, alpha=None):
        """Initializes Dirichlet parameter for f

        Copies alpha, or
        sets f_alpha to be uniform, if alpha == None.

        :param alpha: value to set f_alpha to (15-array)
        :type alpha: numpy.array
        :return: None (sets self.f_alpha)

        """
        if isinstance(alpha, type(None)):
            f_alpha = np.ones_like(self.f_alpha)
        else:
            f_alpha = np.array(alpha).astype('float')
        assert f_alpha.shape == self.f_alpha.shape, \
            'alpha must be numpy.array of shape ' + str(self.f_alpha.shape)
        self.f_alpha = f_alpha

    def init_theta_alpha(self, alpha=None):
        """Initializes Dirichlet parameter for theta

        Copies alpha, or
        sets theta_alpha to be uniform, if alpha == None.

        :param alpha: value to set theta_alpha to (3x15-array)
        :type alpha: numpy.array
        :return: None (sets theta_alpha)

        """
        if isinstance(alpha, type(None)):
            theta_alpha = self.theta_mask.copy()
        else:
            theta_alpha = alpha
        assert theta_alpha.shape == self.theta_alpha.shape, \
            'alpha must be numpy.array of shape ' + str(self.theta_alpha.shape)
        assert np.max(np.abs(theta_alpha * (1.0 - self.theta_mask))) < 1e-30, \
            'alpha must to comply with self.thetamask'
        self.theta_alpha = theta_alpha

    def init_e_array_alpha(self, alpha=100.0):
        """Initializes Dirichlet parameter for  E

        Copies alpha, if it's an array, or
        sets E_alpha[m, i, i, i] = alpha, and all other nonzero elements to 1,
        if alpha is float.

        :param alpha: value to set E_alpha to ((15x3x3x3 array) or float)
        :type alpha: numpy.array or float
        :return: None (sets E_alpha)

        """
        if isinstance(alpha, float):
            e_alpha = np.zeros_like(self.e_alpha)
            e_mask_diag = self.e_array_mask[0, :, :, :]
            for m_idx, m in enumerate(self.modes):
                e_alpha[m_idx, :, :, :] = alpha * e_mask_diag + \
                    (self.e_array_mask[m_idx, :, :, :] - e_mask_diag)
        else:
            e_alpha = np.array(alpha).astype('float')
        assert e_alpha.shape == self.e_alpha.shape, \
            'alpha must be float or numpy.array of shape ' + \
            str(self.e_alpha.shape)
        assert np.max(np.abs(e_alpha * (1.0 - self.e_array_mask))) < 1e-30, \
            'alpha must to comply with self.Emask'
        self.e_alpha = e_alpha

    def _calculate_ev_log_f(self, f_alpha):
        """Returns the expected logarithm of f

        Using the formula in section B.22 of Notes.pdf

        :param f_alpha: Dirichlet parameter for f (15-array)
        :type f_alpha: numpy.array
        :return: ExpValue(log(f)), of shape (15,)
        :rtype: numpy.array

        """
        term1 = digamma(f_alpha)
        term2 = digamma(np.einsum('g->', f_alpha))
        term2_extend = term2 * np.ones(len(self.vfg))
        return term1 - term2_extend

    def _calculate_ev_log_theta(self, theta_alpha, epsilon=1e-30):
        """Returns the expected logarithm of theta (

        Using the formula in section B.23 of Notes.pdf

        :param theta_alpha: Dirichlet parameter for theta (3x15-array)
        :type theta_alpha: numpy.array
        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: ExpValue(log(theta)), of shape (3,15)
        :rtype: numpy.array

        """
        theta_alpha_nonzero = theta_alpha + (1 - self.theta_mask) * epsilon
        term1 = digamma(theta_alpha_nonzero)
        term2 = digamma(np.einsum('cm->c', theta_alpha_nonzero))
        term2_extend = np.einsum('c,m->cm', term2, np.ones(len(self.modes)))
        return term1 - term2_extend

    def _calculate_ev_log_e(self, e_alpha, epsilon=1e-30):
        """Returns the expected logarithm of E

        Using the formula in section B.24 of Notes.pdf

        :param e_alpha: Dirichlet parameter for E (15x3x3x3-array)
        :type e_alpha: numpy.array
        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: ExpValue(log(E)), of shape (15,3,3,3)
        :rtype: numpy.array
        """
        e_alpha_nonzero = e_alpha + (1 - self.e_array_mask) * epsilon
        term1 = digamma(e_alpha_nonzero)
        term2 = digamma(np.einsum('mijk->mi', e_alpha_nonzero))
        term2_extend = np.einsum('mi,jk->mijk',
                                 term2,
                                 np.ones((len(self.ig), len(self.ig))))
        return term1 - term2_extend

    @staticmethod
    def _dirichlet_lognorm(alpha, mask=None):
        """Calculates the logarithm of the normalization constant
        of a Dirichlet distribution

        Only the components of alpha are considered which are which
        correspond 1.0 entries of mask. Entries where mask is 0.0,
        are skipped by the summation.

        :param alpha: Dirichlet parameter vector
        :type alpha: numpy.array
        :param mask: entries in the same shape as alpha,
                     indicating whether the corresponding alpha entry
                     is identically zero (mask == 0.0) or not (mask == 1.0)
        :type mask: numpy.array
        :return: logGamma(sum alpha) - sum logGamma(alpha)
        :rtype: float

        """
        if not isinstance(mask, type(None)):
            alpha = alpha[mask > 0.5]
        term1 = gammaln(np.sum(alpha))
        term2 = np.sum(gammaln(alpha))
        return term1 - term2

    def _calculate_variational_lower_bound(self, epsilon=1e-30):
        """Calculates the variational lower bound of the model likelihood.

        According to section B.4 in Notes.pdf

        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: variational lower bound
        :rtype: float

        """
        len_g12 = len(self.fg)
        ev_log_prob_x = np.einsum(
            'GH,GHgm,ijkgGH,mijk->',
            self.n_array, self.r, self.k_array, self.ev_log_e)
        ev_log_f_extend = np.einsum(
            'g,GHm->GHgm',
            self.ev_log_f, np.ones((len_g12, len_g12, len(self.modes))))
        t = np.einsum(
            'cg,cm->gm',
            self.q_array, self.ev_log_theta)
        ev_log_t_extend = np.einsum(
            'gm,GH->GHgm',
            t, np.ones((len_g12, len_g12)))
        log_r = np.log(self.r + epsilon * (1.0 - self.r_array_mask))
        terms = ev_log_f_extend + ev_log_t_extend - log_r
        ev_log_prob_z = np.einsum(
            'GH,GHgm,GHgm->',
            self.n_array, self.r, terms)
        ev_log_prob_f = self._dirichlet_lognorm(self.f_alpha_prior) \
            - self._dirichlet_lognorm(self.f_alpha) \
            + np.einsum(
                'g,g->',
                self.f_alpha_prior - self.f_alpha, self.ev_log_f)
        ev_log_prob_theta = 0
        for c_idx, c in enumerate(self.groups):
            beta0_c = self.theta_alpha_prior[c_idx, :]
            beta_c = self.theta_alpha[c_idx, :]
            ev_log_theta_c = self.ev_log_theta[c_idx, :]
            mask = self.theta_mask[c_idx, :]
            ev_log_prob_theta_c = self._dirichlet_lognorm(beta0_c, mask=mask) \
                - self._dirichlet_lognorm(beta_c, mask=mask) \
                + np.einsum(
                    'm,m->',
                    beta0_c - beta_c, ev_log_theta_c)
            ev_log_prob_theta += ev_log_prob_theta_c
        ev_log_prob_e = 0
        for m_idx, m in enumerate(self.modes):
            for i_idx, i in enumerate(self.ig):
                gamma0_mi = self.e_alpha_prior[m_idx, i_idx, :, :]
                gamma_mi = self.e_alpha[m_idx, i_idx, :, :]
                ev_log_e_mi = self.ev_log_e[m_idx, i_idx, :, :]
                mask = self.e_array_mask[m_idx, i_idx, :, :]
                ev_log_prob_e_mi = \
                    self._dirichlet_lognorm(gamma0_mi, mask=mask) \
                    - self._dirichlet_lognorm(gamma_mi, mask=mask) \
                    + np.einsum(
                        'jk,jk->',
                        gamma0_mi - gamma_mi, ev_log_e_mi)
                ev_log_prob_e += ev_log_prob_e_mi
        return \
            ev_log_prob_x + \
            ev_log_prob_z + \
            ev_log_prob_f + \
            ev_log_prob_theta + \
            ev_log_prob_e

    def _calculate_r(self, ev_log_f, ev_log_theta, ev_log_e):
        """Calculates responsibilities.

        I.e. R[G1, G2, g, m], which is the estimate of P(g, m | G1, G2),
        from the expected logarithms of f, theta, E.

        :param ev_log_f: Dirichlet parameter for f (15-array)
        :type ev_log_f: numpy.array
        :param ev_log_theta: Dirichlet parameter for theta (3x15-array)
        :type ev_log_theta: numpy.array
        :param ev_log_e: Dirichlet parameter for E (15x3x3x3-array)
        :type ev_log_e: numpy.array
        :return: array of responsibilities, R[G1,G2,g,m]  (27x27x15x15)
        :rtype: numpy. array

        """
        w = np.exp(ev_log_f)
        t = self.t_array_mask * \
            np.exp(np.einsum('cm,cg->gm', ev_log_theta, self.q_array))
        m_array = self.m_array_mask * \
            np.exp(np.einsum('mijk,ijkgGH->gmGH', ev_log_e, self.k_array))
        r_tilde = np.einsum('g,gm,gmGH->GHgm', w, t, m_array)
        norm = np.einsum('GHgm->GH', r_tilde)
        return np.einsum('GHgm,GH->GHgm', r_tilde, 1.0 / norm)

    def _e_step(self):
        """Performs the E-step of the iteration

        Described in Appendix B.2 in Notes.pdf

        :return: None (updates self.ev_log_[f/theta/e] values and self.r)

        """
        self.ev_log_f = self._calculate_ev_log_f(self.f_alpha)
        self.ev_log_theta = self._calculate_ev_log_theta(self.theta_alpha)
        self.ev_log_e = self._calculate_ev_log_e(self.e_alpha)
        self.r = self._calculate_r(
            self.ev_log_f, self.ev_log_theta, self.ev_log_e)

    def _new_f_alpha(self, r):
        """Returns the next value of the Dirichlet parameter of f

        According to Eq. B.17 in Notes.pdf

        :param r: current value of R (27x27x15x15 array)
        :type: numpy.array
        :return: new value of f, of shape (15,)
        :rtype: numpy.array

        """
        n_times_r = np.einsum('GH,GHgm->g', self.n_array, r)
        return self.f_alpha_prior + n_times_r

    def _new_theta_alpha(self, r):
        """Returns the next value of the Dirichlet parameter of theta

        According to Eq. B.18 in Notes.pdf

        :param r: current value of R (27x27x15x15 array)
        :type: numpy.array
        :return: new value of f, of shape (3,15)
        :rtype: numpy.array

        """
        n_times_r_times_q = self.theta_mask * \
            np.einsum('GH,GHgm,cg->cm', self.n_array, r, self.q_array)
        return self.theta_alpha_prior + n_times_r_times_q

    def _new_e_alpha(self, r):
        """Returns the next value of the Dirichlet parameter of E

        According to Eq. B.19 in Notes.pdf

        :param r: current value of R (27x27x15x15 array)
        :type: numpy.array
        :return: new value of f, of shape (15,3,3,3)
        :rtype: numpy.array

        """
        n_times_r_times_k = self.e_array_mask * \
            np.einsum('GH,GHgm,ijkgGH->mijk', self.n_array, r, self.k_array)
        return self.e_alpha_prior + n_times_r_times_k

    def _m_step(self):
        """Performs the M-step of the iteration

        Described in Appendix B.1 in Notes.pdf

        :return: None (updates f_alpha, theta_alpha, e_alpha)

        """
        self.f_alpha = self._new_f_alpha(self.r)
        self.theta_alpha = self._new_theta_alpha(self.r)
        self.e_alpha = self._new_e_alpha(self.r)

    def fit(self,
            threshold=0.1,
            iter_max=int(1e5),
            report_every=10,
            verbose=True):
        """Fits the model with variational EM method

        Initializes f_alpha, theta_alpha, e_alpha values,
        and performs E- and M-steps, until non of the alpha
        parameters change more than threshold between consecutive iterations.

        :param threshold: convergence threshold for alpha parameters
        :type threshold: float
        :param iter_max: maximum number of iterations
        :type iter_max: int
        :param report_every: period of reporting intermediate states
        :type report_every: int
        :param verbose: if True, it prints to stdout
        :type verbose: bool
        :return: list of intermediate results
                 (iter number,
                 variational lower bound,
                 f_alpha,
                 theta_alpha,
                 E_alpha)
        :rtype: list

        """
        it = 0
        iterations = []
        max_diff = 2 * threshold
        try:
            while max_diff >= threshold and it < iter_max:
                self._e_step()  # update w, t, M, R
                if it % report_every == 0:
                    var_lower_bound = self._calculate_variational_lower_bound()
                    iterations.append((it,
                                       var_lower_bound,
                                       self.f_alpha,
                                       self.theta_alpha,
                                       self.e_alpha))
                    if verbose and it % 1000 == 0:
                        print str(it) + \
                            ' iterations done, ' +\
                            'log-likelihood lower bound: ' + \
                            str(var_lower_bound)
                f_alpha_prev = self.f_alpha.copy()
                theta_alpha_prev = self.theta_alpha.copy()
                e_alpha_prev = self.e_alpha.copy()

                self._m_step()  # update f, theta, E
                max_diff_f = np.max(np.abs(self.f_alpha - f_alpha_prev))
                max_diff_theta = \
                    np.max(np.abs(self.theta_alpha - theta_alpha_prev))
                max_diff_e = np.max(np.abs(self.e_alpha - e_alpha_prev))
                max_diff = np.max([max_diff_f, max_diff_theta, max_diff_e])

                it += 1
            if it >= iter_max:
                print 'Warning: iter_max reached, EM did not reach target'
        except KeyboardInterrupt:
            print 'Warning: User interrupt, EM did not reach target.'
            pass
        return iterations

    def sample_n_array_complete(self, total_samples, r=None):
        """Draws samples from the estimated posterior of N_complete

        Samples are drawn from the variational estimate of the posterior
        defined by the current alpha parameters of the model

        :param total_samples: number of samples
        :type total_samples: int
        :param r: (Optional) responsibility matrix to be used
                  instead of the one calculated from model parameters
        :return: None (appends to n_array_complete_samples)

        """
        len_g12 = len(self.fg)
        len_g = len(self.vfg)
        len_m = len(self.modes)
        n_array = self.n_array

        if isinstance(r, type(None)):
            r = self.r

        for sample_idx in range(total_samples):
            n_array_complete = np.zeros((len_g12, len_g12, len_g, len_m))
            for G1_idx in range(len_g12):
                for G2_idx in range(len_g12):
                    n = n_array[G1_idx, G2_idx]
                    p = r[G1_idx, G2_idx, :, :].flatten()
                    n_array_complete[G1_idx, G2_idx, :, :] = \
                        np.random.multinomial(n, p).reshape((len_g, len_m))
            self.n_array_complete_samples.append(n_array_complete)
