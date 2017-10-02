import numpy as np
from model import GeckModel


class GeckSolverGibbs(GeckModel):
    """
    Gibbs sampling solution described in Appendix C of Notes.pdf

    :ivar n_array: aggregate trio genotype counts (27x27-array)
    :ivar f_alpha_prior: Dirichlet parameter of the prior of f
    :ivar theta_alpha_prior: Dirichlet parameter of the prior of theta
    :ivar e_array_alpha_prior: Dirichlet parameter of the prior of E

    :ivar parameter_samples: list of (f, theta, E) samples
    :ivar n_array_complete_samples:
        list of N_complete samples, the prediction of the model
        (list of 27x27x15x15-arrays)

    """
    def __init__(self):
        GeckModel.__init__(self)

        self.n_array = (-1) * np.ones((len(self.fg), len(self.fg)))

        self.f_alpha_prior = 1.0 * np.ones(len(self.vfg))
        self.theta_alpha_prior = 1.0 * self.theta_mask.copy()
        self.e_array_alpha_prior = 1.0 * self.e_array_mask.copy()

        self.parameter_samples = []
        self.n_array_complete_samples = []

    def import_data(self, geck_data):
        """Gets aggregate data from a GeckData instance

        Gets aggregate data from a GeckData instance,
        while making sure the ordering matches with self.fg

        :param geck_data: input data
        :type geck_data: GeckData
        :return: None (sets self.N)

        """
        self.n_array = geck_data.get_data_matrix(self.fg)

    def _sample_n_array_complete(self, n_array, f, theta, e_array):
        """Draws a single sample from the P(N_complete | N, f, theta, E)

        Different N_complete[G1, G2, :, :] slices of N_complete
        are sampled from the corresponding multinomial distributions,
        dependent on the model parameters (f, theta, E),
        and the observed counts N[G1, G2]

        :param n_array: observed data (27x27)
        :type n_array: numpy.array
        :param f: trio frequency (15)
        :type f: numpy.array
        :param theta: fraction of variants affected by each error mode (3x15)
        :type theta: numpy.array
        :param e_array: error rates (15x3x3x3)
        :type e_array: numpy.array
        :return: n_array_complete (27x27x15x15)
        :rtype: numpy.array

        """
        prob = self.calculate_distribution(f, theta, e_array)
        norm = np.einsum('GHgm->GH', prob)
        r = np.einsum('GHgm,GH->GHgm', prob, 1.0 / norm)
        len_g12 = len(self.fg)
        len_g = len(self.vfg)
        len_m = len(self.modes)
        n_array_complete = np.zeros((len_g12, len_g12, len_g, len_m))
        for G1_idx in range(len_g12):
            for G2_idx in range(len_g12):
                n = n_array[G1_idx, G2_idx]
                p = r[G1_idx, G2_idx, :, :].flatten()
                n_array_complete[G1_idx, G2_idx, :, :] = \
                    np.random.multinomial(n, p).reshape((len_g, len_m))
        return n_array_complete

    def _sample_f(self, n_array_complete):
        """Draws a single sample from P(f | N_complete)

        Frequencies are sampled from a Dirichlet distribution,
        that depends on the prior parameters and N_complete

        :param n_array_complete: complete data (27x27x15x15)
        :type n_array_complete: numpy.array
        :return: sampled frequency array (15,)
        :rtype: numpy.array

        """
        alpha = self.f_alpha_prior + np.einsum('GHgm->g', n_array_complete)
        f = np.random.dirichlet(alpha)
        return f

    def _sample_theta(self, n_array_complete):
        """Draws a single sample from P(theta | N_complete)

        Each theta[s,:] slice of the error model fractions
        is sampled from a Dirichlet distribution, which
        depends on N_complete and the prior parameters

        :param n_array_complete: complete data (27x27x15x15)
        :type n_array_complete: numpy.array
        :return: error model fractions (3x5)
        :rtype: numpy.array

        """
        lenm = len(self.modes)
        lenc = len(self.groups)
        beta = self.theta_alpha_prior + \
            np.einsum('GHgm,cg->cm', n_array_complete, self.q_array)
        theta = np.zeros((lenc, lenm))
        for c_idx in range(lenc):
            theta[c_idx, :] = np.random.dirichlet(beta[c_idx, :])
        return theta

    def _sample_e_array(self, n_array_complete):
        """Draws a single sample from P(E | N_complete)

        Each E[m,i,:,:] slice of error rates is sampled from
        a Dirichlet distribution, which depends on N_complete
        and the prior parameters.
        It also makes sure that E[m,i,i,i] is the maximum entry
        in each E[m,i,:,:] matrix.

        :param n_array_complete: complete data (27x27x15x15)
        :type n_array_complete: numpy.array
        :return: error rate array (15,3,3,3)
        :rtype: numpy.array

        """
        len_i = len(self.ig)
        len_m = len(self.modes)
        gamma = self.e_array_alpha_prior + \
            np.einsum('GHgm,ijkgGH->mijk', n_array_complete, self.k_array)

        e_array = np.zeros((len_m, len_i, len_i, len_i))
        for m_idx in range(len_m):
            for i_idx in range(len_i):
                gamma_mi_vec = gamma[m_idx, i_idx, :, :].flatten()
                e_array[m_idx, i_idx, :, :] = \
                    np.random.dirichlet(gamma_mi_vec).reshape((len_i, len_i))
                if np.argmax(e_array[m_idx, i_idx, :, :]) != 4 * i_idx:
                    e_array[m_idx, i_idx, i_idx, i_idx] = \
                        np.max(e_array[m_idx, i_idx, :, :])
                    e_array[m_idx, i_idx, :, :] /=  \
                        np.sum(e_array[m_idx, i_idx, :, :])
        return e_array

    def run_sampling(self,
                     f0=None,
                     theta0=None,
                     e_array_0=None,
                     f_samples=None,
                     theta_samples=None,
                     e_array_samples=None,
                     burnin=0,
                     every=1,
                     iterations=int(1e4),
                     verbose=True):
        """Performs Gibbs sampling by alternating

         1. sampling (f, theta, E) from their Dirichlet distributions,
            which are conditioned on N_complete
         2. sampling N_complete from a series of multinomials,
            which are conditioned on (f, theta, E)

        :param f0: starting value for f (15-array)
        :type f0: numpy.array
        :param theta0: starting value for theta (3x15-array)
        :type theta0: numpy.array
        :param e_array_0: stating value for E0 (15x3x3x3-array)
        :type e_array_0: numpy.array
        :param f_samples: if set, f is re-sampled
                          from this list uniformly (list of 15-array)
        :type f_samples: list
        :param theta_samples: if set, theta is re-sampled
                              from this list uniformly (list of 3x15-array)
        :type theta_samples: list
        :param e_array_samples: if set, E is re-sampled
                          from this list uniformly (list of 15x3x3x3-array)
        :type e_array_samples: list
        :param burnin: number of iterations to be discarded from the beginning
        :type burnin: int
        :param every: period of saving a sample
        :type every: int
        :param iterations: total number of raw iterations after the burn-in
                           (if every > 1, the actually obtained
                           number of samples is iterations / every)
        :type iterations: int
        :param verbose: if True, print status messages to stdout
        :type verbose: bool
        :return: None (appends to self.parameter_samples
                      and self.n_array_complete_samples)

        """
        if isinstance(f0, type(None)):
            f0 = self.random_f()
        if isinstance(theta0, type(None)):
            theta0 = self.random_theta()
        if isinstance(e_array_0, type(None)):
            e_array_0 = self.random_e_array()

        resample_f = (not isinstance(f_samples, type(None)))
        resample_theta = (not isinstance(theta_samples, type(None)))
        resample_e_array = (not isinstance(e_array_samples, type(None)))

        len_f_samples = 0
        len_theta_samples = 0
        len_e_array_samples = 0
        if resample_f:
            len_f_samples = len(f_samples)
        if resample_theta:
            len_theta_samples = len(theta_samples)
        if resample_e_array:
            len_e_array_samples = len(e_array_samples)

        f = f_samples[np.random.randint(0, len_f_samples)] \
            if resample_f else f0
        theta = theta_samples[np.random.randint(0, len_theta_samples)] \
            if resample_theta else theta0
        e_array = e_array_samples[np.random.randint(0, len_e_array_samples)] \
            if resample_e_array else e_array_0

        n_array = self.n_array

        try:
            for it in range(burnin):
                n_array_complete = \
                    self._sample_n_array_complete(n_array, f, theta, e_array)
                f = f_samples[np.random.randint(0, len_f_samples)] \
                    if resample_f else self._sample_f(n_array_complete)
                theta = \
                    theta_samples[np.random.randint(0, len_theta_samples)] \
                    if resample_theta else self._sample_theta(
                        n_array_complete)
                e_array = \
                    e_array_samples[np.random.randint(0,
                                                      len_e_array_samples)] \
                    if resample_e_array else self._sample_e_array(
                        n_array_complete)
                if verbose and it % 1000 == 0:
                    print str(it) + ' iterations done, burning in'
        except KeyboardInterrupt:
            print 'Warning: User interrupt. No samples collected. Exiting...'
            exit(1)
        if verbose and burnin > 0:
            print str(burnin) + ' iterations burned'

        try:
            for it in range(iterations):
                n_array_complete = \
                    self._sample_n_array_complete(n_array, f, theta, e_array)
                f = \
                    f_samples[np.random.randint(0, len_f_samples)] \
                    if resample_f \
                    else self._sample_f(n_array_complete)
                theta = \
                    theta_samples[np.random.randint(0, len_theta_samples)] \
                    if resample_theta \
                    else self._sample_theta(n_array_complete)
                e_array = \
                    e_array_samples[np.random.randint(0,
                                                      len_e_array_samples)] \
                    if resample_e_array \
                    else self._sample_e_array(n_array_complete)
                if it % every == 0:
                    self.parameter_samples.append((f, theta, e_array))
                    self.n_array_complete_samples.append(n_array_complete)
                if verbose and it % 1000 == 0:
                    print str(it) + ' iterations done, ' + str(it / every) + \
                        ' samples collected'
        except KeyboardInterrupt:
            print 'Warning: User interrupt.'
        if verbose:
            print str(iterations) + ' iterations done, ' + \
                str(iterations / every) + ' samples collected'
