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
from model import GeckModel


class GeckSolverEM(GeckModel):
    """Expectation Maximization solution described in Appendix A in Notes.pdf

    :ivar n_array: aggregate trio genotype counts (27x27-array)
    :ivar f_array: genotype trio frequencies (15-array)
    :ivar theta_array: fraction of variants affected by
                       different error modes, in the 2 groups (3x15-array)
    :ivar e_array: genotyping error rates in
                   the 5 different modes (15x3x3x3-array)

    :ivar f_min: hard lower threshold for all components of f (float)
    :ivar theta_min: hard lower threshold for all components of theta (float)
    :ivar e_array_min: hard lower threshold for non-zero components of
                       E (defined by self.e_array_mask) (float)

    :ivar w_array: :math:`P(g | f)` (15-array)
    :ivar t_array: :math:`P(m | g, \\theta)` (15x15-array)
    :ivar m_array: :math:`P(G1, G2 | g, m)` (15x15x27x27-array)
    :ivar r_array: :math:`P(g,m | G1, G2)` (27x27x15x15-array)

    :ivar n_array_complete_samples: list of n_array_complete samples,
        the prediction of the model (list of 27x27x15x5-arrays)

    """
    def __init__(self):
        GeckModel.__init__(self)

        self.n_array = (-1) * np.ones((len(self.fg), len(self.fg)))

        self.f_array = (-1) * np.ones(len(self.vfg))
        self.theta_array = (-1) * np.ones((len(self.groups), len(self.modes)))
        self.e_array = (-1) * np.ones((len(self.modes), len(self.ig),
                                       len(self.ig), len(self.ig)))
        self.f_min = 0.0
        self.theta_min = 0.0
        self.e_min = 0.0

        self.w_array = (-1) * np.ones(len(self.vfg))
        self.t_array = (-1) * np.ones((len(self.vfg), len(self.modes)))
        self.m_array = (-1) * np.ones((len(self.vfg), len(self.modes),
                                       len(self.fg), len(self.fg)))
        self.r_array = (-1) * np.ones((len(self.fg), len(self.fg),
                                       len(self.vfg), len(self.modes)))

        self.n_array_complete_samples = []

    def import_data(self, geck_data):
        """Gets aggregate data from a GeckData instance,

        while making sure the ordering matches with self.fg

        :param geck_data: input data, i.e. joint histogram of genotyping trio
                          counts
        :type geck_data: GeckData
        :return: None (sets self.f)

        """
        self.n_array = geck_data.get_data_matrix(self.fg)

    def init_f(self, f0=None):
        """ Initializes true frequencies,

        either by copying f0, or
        if f0 == None, setting it to be uniform

        :param f0: floats of shape (15,), summing up to 1.0
        :type f0: numpy.array
        :return: None (sets self.f_array)

        """
        if isinstance(f0, type(None)):
            f = np.ones_like(self.f_array)
            f /= np.sum(f)
        else:
            f = np.array(f0).astype('float')
        assert f.shape == self.f_array.shape, \
            'f0 must be numpy.array of shape ' + str(self.f_array.shape)
        assert np.abs(np.einsum('g->', f) - 1) < 1e-5, \
            'values of f0 must sum to 1.0'
        self.f_array = f

    def init_theta(self, theta0=None):
        """Initializes parameter :math:`\theta`,

        either by copying theta0, or
        if theta0 == None, setting it to be uniform

        :param theta0: fraction of variants in each error mode (shape (3,15)),
                       each row summing up to 1.0
        :type theta0: numpy.array
        :return: None (sets self.theta_array)

        """
        if isinstance(theta0, type(None)):
            theta = self.theta_mask.copy()
            norm = np.einsum('cm->c', theta)
            theta = np.einsum('cm,c->cm', theta, 1.0 / norm)
        else:
            theta = theta0
        assert theta.shape == self.theta_array.shape
        assert np.max(np.abs(theta * (1.0 - self.theta_mask))) < 1e-30, \
            'theta0 must to comply with thetamask'
        assert np.max(np.abs(np.einsum('cm->c', theta) - 1.0)) < 1e-5, \
            'theta0 must be normalized for all values of index 0'
        self.theta_array = theta

    def init_e_array(self, e0=0.01):
        """Initializes parameter E,

        either by copying E0,
        or, if E0 is float, setting E[m, i, i, i] = 1,
        and all other nonzero elements to 0.01, and normalizing

        :param e0: float OR
                   numpy array of floats of shape (15,3,3,3),
                   each [m,i,:,:] block summing up to 1.0
        :type e0: float or numpy.array
        :return: None (sets self.E)

        """
        if isinstance(e0, type(0.01)):
            e_array = np.zeros_like(self.e_array)
            e_array_mask_00 = self.e_array_mask[0, :, :, :]
            for m_idx, m in enumerate(self.modes):
                e_array[m_idx, :, :, :] = e_array_mask_00 + \
                    e0 * (self.e_array_mask[m_idx, :, :, :] - e_array_mask_00)
            norm = np.einsum('mijk->mi', e_array)
            e_array = np.einsum('mijk,mi->mijk', e_array, 1.0 / norm)
        else:
            e_array = np.array(e0).astype('float')
        assert e_array.shape == self.e_array.shape, \
            'E0 must be numpy.array of shape ' + str(self.e_array.shape)
        assert np.max(np.abs(e_array * (1.0 - self.e_array_mask))) < 1e-30, \
            'E0 must to comply with Emask'
        assert np.max(np.abs(np.einsum('mijk->mi', e_array) - 1.0)) < 1e-5, \
            'E0 must be normalized for all values of index 0 and 1'
        self.e_array = e_array

    @staticmethod
    def _calculate_w(f):
        """Calculates :math:`P(g | f)`

        The probability of a genotype trio g, given frequencies f

        :param f: frequencies, of shape (15,)
        :type f: numpy array
        :return: w, shape (15,)
        :rtype: numpy.array

        """
        return f

    def _calculate_t(self, theta, epsilon=1e-30):
        """Calculates :math:`P(m | g, \\theta)`

        The probability of an error model m, given the trio genotype g,
        and the error model fractions theta.

        :param theta: error model fraction, of shape (3, 15)
        :type theta: numpy.array
        :return: t, of shape (15, 15)
        :rtype: numpy.array

        """
        logtheta = np.log(theta + epsilon * (1 - self.theta_mask))
        q_array = self.q_array
        return self.t_array_mask * \
            np.exp(np.einsum('cm,cg->gm', logtheta, q_array))

    def _calculate_m_array(self, e_array, epsilon=1e-30):
        """Calculates :math:`P(G1, G2 | g, m, E)`

        The probability of a called pair of genotype trios (G1,G2),
        given the true genotype trio g, error model m,
        and error rate array E

        :param e_array: error rates, of shape (15, 3, 3, 3)
        :type e_array: numpy array
        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: M, of shape (15, 15, 27, 27)
        :rtype: numpy.array

        """
        log_e_array = np.log(e_array + epsilon * (1 - self.e_array_mask))
        k_array = self.k_array
        return self.m_array_mask * \
            np.exp(np.einsum('mijk,ijkgGH->gmGH', log_e_array, k_array))

    @staticmethod
    def _calculate_prob_gmg1g2(w, t, m_array):
        """Calculates :math:`P(g, m, G1, G2 | f, \\theta, E)`

        The probability of observed (G1, G2) and hidden (g,m) variables,
        given the conditional probabilities

        :param w: P(g), of shape (15,)
        :type w: numpy.array
        :param t: P(m|g), of shape (15, 15)
        :type t: numpy.array
        :param m_array: P(G1,G2|g,m) of shape (15, 15, 27, 27)
        :type m_array: numpy.array
        :return: P(g,m,G1,G2), of shape (15, 15, 27, 27)
        :rtype: numpy.array

        """
        return np.einsum('g,gm,gmGH->gmGH', w, t, m_array)

    @staticmethod
    def _calculate_prob_g1g2(w, t, m_array):
        """Calculates :math:`P(G1, G2 | f, \\theta, E)`

        The probability of observed (G1, G2) variables,
        given the conditional probabilities

        :param w: P(g), of shape (15,)
        :type w: numpy.array
        :param t: P(m|g), of shape (15, 15)
        :type t: numpy.array
        :param m_array: P(G1,G2|g,m) of shape (15, 15, 27, 27)
        :type m_array: numpy.array
        :return: P(G1,G2), of shape (27, 27)
        :rtype: numpy.array

        """
        return np.einsum('g,gm,gmGH->GH', w, t, m_array)

    def _calculate_r_array(self, w, t, m_array):
        """Calculates the responsibility P(g, m | G1, G2)

        The probability of hidden variables (g,m) conditioned on the
        observed variables (G1, G2)

        :param w: P(g), of shape (15,)
        :type w: numpy.array
        :param t: P(m|g), of shape (15, 15)
        :type t: numpy.array
        :param m_array: P(G1,G2|g,m) of shape (15, 15, 27, 27)
        :type m_array: numpy.array
        :return: P(g,m | G1,G2), of shape (27, 27, 15, 15)
        :rtype: numpy.array

        """
        prob_gmg1g2 = self._calculate_prob_gmg1g2(w, t, m_array)
        prob_g1g2 = np.einsum('gmGH->GH', prob_gmg1g2)
        return np.einsum('gmGH,GH->GHgm', prob_gmg1g2, 1.0 / prob_g1g2)

    def _calculate_loglikelihood(self, w, t, m_array):
        """Calculates the Log-likelihood of the model

        The probability that the observed data, N[G1,G2]
        was produced by the model with current parameter settings.

        :param w: P(g), of shape (15,)
        :type w: numpy.array
        :param t: P(m|g), of shape (15, 15)
        :type t: numpy.array
        :param m_array: P(G1,G2|g,m) of shape (15, 15, 27, 27)
        :type m_array: numpy.array
        :return: log P(N[G1,G2])
        :rtype: float

        """
        log_prob_g1g2 = np.log(self._calculate_prob_g1g2(w, t, m_array))
        return np.einsum('GH,GH->', self.n_array, log_prob_g1g2)

    def _e_step(self):
        """Performes and E-step of Expectation Maximization

        More details can be found in Appendix A.2 in Notes.pdf

        :return: None (updates self.w, t, M, R)

        """
        self.w_array = self._calculate_w(self.f_array)
        self.t_array = self._calculate_t(self.theta_array)
        self.m_array = self._calculate_m_array(self.e_array)
        self.r_array = self._calculate_r_array(self.w_array,
                                               self.t_array,
                                               self.m_array)

    def _new_f(self, r_array):
        """Calculates a new frequency vector

        according to Eq. A.3 from Appendix A.1

        :param r_array: responsibilities, of shape (27, 27, 15, 15)
        :type r_array: numpy.array
        :return: new frequencies, f of shape (15,)
        :rtype: numpy.array

        """
        n_times_r = np.einsum('GH,GHgm->g', self.n_array, r_array)
        norm = np.einsum('g->', n_times_r)
        f = n_times_r / float(norm)
        f[f < self.f_min] = self.f_min
        norm = np.einsum('g->', f)
        return f / float(norm)

    def _new_theta(self, r_array):
        """Calculates a new error model fractions

        according to Eq. A.4 from Appendix A.1

        :param r_array: responsibilities, of shape (27, 27, 15, 15)
        :type r_array: numpy.array
        :return: new fractions, theta of shape (3,15)
        :rtype: numpy.array

        """
        n_times_r_times_q = self.theta_mask * \
            np.einsum('GH,GHgm,cg->cm', self.n_array, r_array, self.q_array)
        norm = np.einsum('cm->c', n_times_r_times_q)
        theta = np.einsum('cm,c->cm', n_times_r_times_q, 1.0 / norm)
        theta[(theta < self.theta_min) & (self.theta_mask > 0.5)] \
            = self.theta_min
        norm = np.einsum('cm->c', theta)
        theta = np.einsum('cm,c->cm', theta, 1.0 / norm)
        return theta

    def _new_e_array(self, r_array, epsilon=1e-30):
        """Calculates a new error rates

        according to Eq. A.5 from Appendix A.1

        :param r_array: responsibilities, of shape (27, 27, 15, 15)
        :type r_array: numpy.array
        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: new error rates, E of shape (15,3,3,3)
        :rtype: numpy.array

        """
        n_times_r_times_k = self.e_array_mask * \
            np.einsum('GH,GHgm,ijkgGH->mijk',
                      self.n_array,
                      r_array,
                      self.k_array)
        norm = np.einsum(
            'mijk->mi',
            n_times_r_times_k + (1.0 - self.e_array_mask) * epsilon)
        e_array = np.einsum('mijk,mi->mijk', n_times_r_times_k, 1.0 / norm)
        e_array[(e_array < self.e_min) & (self.e_array_mask > 0.5)] = \
            self.e_min
        norm = np.einsum('mijk->mi', e_array)
        e_array = np.einsum('mijk,mi->mijk', e_array, 1.0 / norm)
        return e_array

    def _m_step(self):
        """Performs the M-step of Expectation Maximization

        according to the details in Appendix A.1 in Notes.pdf

        :return: None (updates self.f, theta, E)

        """
        self.f_array = self._new_f(self.r_array)
        self.theta_array = self._new_theta(self.r_array)
        self.e_array = self._new_e_array(self.r_array)

    def fit(self,
            threshold=1e-8,
            iter_max=int(1e5),
            report_every=10,
            verbose=True):
        """Performs fitting the model with EM method

        Initializes variables and alternates E- and M-steps,
        until non of the components of f, theta, E changes more than
        threshold in a single iteration.

        :param threshold: if difference between consecutive
                          iterations of all parameters is <= threshold
                          -> declare convergence, stop EM
        :type threshold: float
        :param iter_max: maximum total number of iterations
        :type iter_max: int
        :param report_every: periodicity of recording intermediate results
                             (log-likelihood, f, theta, E)
        :type report_every: int
        :param verbose: if True, prints status to stdout periodically
        :type verbose: bool
        :return: list of (log-likelihood, f, theta, E)
        :rtype: list

        """

        it = 0
        self._e_step()  # update w, t, M, R
        log_likelihood = self._calculate_loglikelihood(
            self.w_array, self.t_array, self.m_array)
        iterations = []
        max_diff = 2 * threshold
        try:
            while max_diff >= threshold and it < iter_max:
                self._e_step()  # update w, t, M, R
                if it % report_every == 0:
                    log_likelihood = self._calculate_loglikelihood(
                        self.w_array, self.t_array, self.m_array)
                    iterations.append(
                        (it,
                         log_likelihood,
                         self.f_array,
                         self.theta_array,
                         self.e_array))
                    if verbose and it % 1000 == 0:
                        print str(it) + ' iterations done, log-likelihood: ' \
                            + str(log_likelihood)
                f_prev = self.f_array.copy()
                theta_prev = self.theta_array.copy()
                e_array_prev = self.e_array.copy()

                self._m_step()  # update f, theta, E
                max_diff_f = np.max(np.abs(self.f_array - f_prev))
                max_diff_theta = np.max(np.abs(self.theta_array - theta_prev))
                max_diff_e_array = np.max(np.abs(self.e_array - e_array_prev))
                max_diff = np.max(
                    [max_diff_f, max_diff_theta, max_diff_e_array])

                it += 1
            if verbose:
                print str(it) + ' iterations done, log-likelihood: ' \
                    + str(log_likelihood)
            if it >= iter_max:
                print 'Warning: iter_max reached, EM did not reach target'
        except KeyboardInterrupt:
            print 'Warning: User interrupt, EM did not reach target.'
            pass
        return iterations

    def sample_n_array_complete(self, total_samples, r_array=None):
        """Draws samples from the estimated posterior of N_complete

        Draws (g,m,G1,G2) samples from the multinomial
        model described by the current model parameter (f, theta, E)

        :param total_samples: number of samples
        :type total_samples: int
        :param r_array: responsibility matrix to be used
        :type r_array: numpy.array
        :return: None (appends to Ncomplete_samples)

        """
        len_g12 = len(self.fg)
        len_g = len(self.vfg)
        lenm = len(self.modes)
        n_array = self.n_array

        if isinstance(r_array, type(None)):
            r_array = self.r_array

        for sample_idx in range(total_samples):
            n_array_complete = np.zeros((len_g12, len_g12, len_g, lenm))
            for G1_idx in range(len_g12):
                for G2_idx in range(len_g12):
                    n = n_array[G1_idx, G2_idx]
                    p = r_array[G1_idx, G2_idx, :, :].flatten()
                    n_array_complete[G1_idx, G2_idx, :, :] = \
                        np.random.multinomial(n, p).reshape((len_g, lenm))
            self.n_array_complete_samples.append(n_array_complete)
