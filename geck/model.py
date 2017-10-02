import itertools
import numpy as np


class GeckModel:
    """
    Implements the statistical model described in section 2.4 in Notes.pdf

    :ivar alleles: tuple of alleles (currently fixed to ('0', '1'))
    :ivar ig: individual diploid genotypes (3-array)
    :ivar fg: family genotype trios (27-array)
    :ivar vfg: valid family genotype trios (valid: Mendelian compliant)
               (15-array)
    :ivar groups: labels for genotype subsets,
                  which are affected by the same error profiles (3-tuple)
    :ivar modes: labels for error modes (15-tuple)
    :ivar q_array: indicates which genotype belongs to which group
                   (3x15-array)
    :ivar k_array: array counting how many i->j,k genotype
                   transitions happen in g->G1,G2 (3x3x3x15x27x27-array)
    :ivar theta_mask: indicator, 1.0 if theta is allowed
                      to be non-zero, 0.0 otherwise (3x15-array)
    :ivar e_array_mask: indicator, 1.0 if E is allowed
                        to be non-zero, 0.0 otherwise (15x3x3x3-array)
    :ivar t_array_mask: indicator, 1.0 if t is allowed
                        to be non-zero, 0.0 otherwise (15x15-array)
    :ivar m_array_mask: indicator, 1.0 if M is allowed
                        to be non-zero, 0.0 otherwise (15x5x27x27-array)
    :ivar r_array_mask: indicator, 1.0 if R is allowed
                        to be non-zero, 0.0 otherwise (27x27x15x15-array)

    """

    def __init__(self):
        self.alleles = ('0', '1')
        self.ig = self._init_individual_genotypes()
        self.fg = self._init_family_genotypes()
        self.vfg = self._init_valid_genotypes()

        self.k_array_father, self.k_array_mother, self.k_array_child = \
            self._init_k_array()
        self.k_array = \
            self.k_array_father + self.k_array_mother + self.k_array_child

        self.groups, self.modes, self.q_array, \
            self.theta_mask, self.t_array_mask, \
            self.e_array_mask, self.m_array_mask, self.r_array_mask \
            = self._define_groups_and_modes()

    def _init_individual_genotypes(self):
        """generates diploid genotypes,

        i.e. ('00', '01', '11')

        :return: possible individual genotypes
        :rtype: tuple

        """
        alleles = self.alleles
        ig_set = set([])
        for a1 in alleles:
            for a2 in alleles:
                a = sorted([a1, a2])
                ig_set.add(''.join(a))
        return tuple(sorted(list(ig_set)))

    def _init_family_genotypes(self):
        """generates all (mathematically) possible family genotypes

        i.e. (('00','00','00'), ('00','00','01'), ...)

        :return: possible genotype trios
        :rtype: tuple

        """
        ig = self.ig
        return tuple([gt for gt in itertools.product(ig, ig, ig)])

    def _init_valid_genotypes(self):
        """generates all Mendelian-compliant family genotype

        i.e. (('00','00','00'), ('00','01','00'), ...)

        :return: genotype trios that comply with Mendelian inheritance
        :rtype: tuple

        """
        valid_family_gts = []
        for g in self.fg:
            father = g[0]
            mother = g[1]
            child = g[2]
            if (child[0] in father and child[1] in mother) or \
                    (child[0] in mother and child[1] in father):
                valid_family_gts.append(g)
        return tuple(valid_family_gts)

    def _init_k_array(self):
        """generates three K-arrays, one for each family member

        They indicate if a certain individual genotype transition (i->(j,k))
        happen in a family genotype transition event (g->(G1,G2))

        :return: k-arrays, one for each family member
        :rtype: list

        """
        k_array_list = []
        for person in range(3):
            k_array = np.zeros([len(self.ig), len(self.ig), len(self.ig),
                                len(self.vfg), len(self.fg), len(self.fg)])
            for g_idx, g in enumerate(self.vfg):
                for G1_idx, G1 in enumerate(self.fg):
                    for G2_idx, G2 in enumerate(self.fg):
                        i_idx = self.ig.index(g[person])
                        j_idx = self.ig.index(G1[person])
                        k_idx = self.ig.index(G2[person])
                        k_array[i_idx, j_idx, k_idx, g_idx, G1_idx, G2_idx] \
                            += 1
            k_array_list.append(k_array)
        return k_array_list

    def _define_groups_and_modes(self):
        """generates the three subset of genotypes defines error matrices

        Subsets:
         - 00,00,00
         - 11,11,11
         - neither
        i.e. returns the labels (groups) and the indicator matrix (S)
             where S[c,g] == 1 indicates that genotype g belongs to group c

        Eerror matrices for error modes:
        i.e. returns labels (modes) and masks (Emask, Mmask, Rmask)
             where Emask[m,i,j,k] == 0
                        indicates that E[m,i,j,k] is identically zero
                   Mmask[g,m,G1,G2] == 0
                        indicates that M[g,m,G1,G2] is identically zero
                   Rmask[G1,G2,g,m] == 0
                        indicates that R[G1,G2,g,m] is identically zero

        :return: tuple of
            * groups: tuple of subset names
            * modes: tuple of error mode names
            * q_array: matrix of :math: `Q_{s,g}`
            * theta_mask: 1's at nonzero :math: `\\theta_{s,m}` elements
            * t_array_mask: 1's at nonzero :math: `t_{g,m}` elements
            * e_array_mask: 1's at nonzero :math: `E^{m}_{i,j,k}` elements
            * m_array_mask: 1's at nonzero :math: `M_{g,m,G1, G2}` elements
            * r_array_mask: 1's at nonzero :math: `R_{G1,G2,g,m}` elements
        :rtype: tuple

        """
        groups = ('0 - (00,00,00)',
                  '1 - neither',
                  '2 - (11,11,11)')
        q_array = np.zeros((len(groups), len(self.vfg)))
        for c_idx, c in enumerate(groups):
            for g_idx, g in enumerate(self.vfg):
                if c[0] == '0' and g == ('00', '00', '00'):
                    q_array[c_idx, g_idx] = 1
                elif c[0] == '1' and g not in [('00', '00', '00'),
                                               ('11', '11', '11')]:
                    q_array[c_idx, g_idx] = 1
                elif c[0] == '2' and g == ('11', '11', '11'):
                    q_array[c_idx, g_idx] = 1

        pre_modes = ('00 - no error',
                     '01 - no error by tool 1',
                     '10 - no error by tool 2',
                     '11 - identical errors by the two tools',
                     '12 - uncorrelated errors')
        modes = []
        for c in groups:
            c_tup = c.split(' - ')
            for pm in pre_modes:
                pm_tup = pm.split(' - ')
                m = c_tup[0] + '.' + pm_tup[0] + ' - ' + \
                    c_tup[1] + ', ' + pm_tup[1]
                modes.append(m)
        modes = tuple(modes)

        thetamask = np.zeros((len(groups), len(modes)))
        for c_idx, c in enumerate(groups):
            for m_idx, m in enumerate(modes):
                c_code = m[0]
                if c[0] == c_code:
                    thetamask[c_idx, m_idx] = 1

        epsilon = 1e-30
        logthetamask = np.log(thetamask + (1 - thetamask) * epsilon)
        exp_q_array_log_theta_mask = \
            np.exp(np.einsum('cg,cm->gm', q_array, logthetamask))
        tmask = np.zeros((len(self.vfg), len(modes)))
        tmask[exp_q_array_log_theta_mask > 0.5] = 1

        conditions = {
            '00': lambda true, call1, call2: call1 == true and call2 == true,
            '01': lambda true, call1, call2: call1 == true,
            '10': lambda true, call1, call2: call2 == true,
            '11': lambda true, call1, call2: call1 == call2,
            '12': lambda true, call1, call2: True}

        e_array_mask = np.zeros(
            (len(modes), len(self.ig), len(self.ig), len(self.ig)))
        for m_idx, m in enumerate(modes):
            c_code = m[0]
            pm_code = m[2:4]
            for i_idx, i in enumerate(self.ig):
                if (c_code == '0' and i != '00') or \
                   (c_code == '2' and i != '11'):
                    pm = '00'
                else:
                    pm = pm_code
                for j_idx, j in enumerate(self.ig):
                    for k_idx, k in enumerate(self.ig):
                        if conditions[pm](i, j, k):
                            e_array_mask[m_idx, i_idx, j_idx, k_idx] = 1

        epsilon = 1e-30
        log_e_array_mask = np.log(e_array_mask + (1 - e_array_mask) * epsilon)
        exp_k_array_log_e_array_mask = \
            np.exp(np.einsum('ijkgGH,mijk->gmGH',
                             self.k_array,
                             log_e_array_mask))
        m_array_mask = np.zeros(
            (len(self.vfg), len(modes), len(self.fg), len(self.fg)))
        m_array_mask[exp_k_array_log_e_array_mask > 0.5] = 1

        r_array_mask = np.einsum('gm,gmGH->GHgm', tmask, m_array_mask)

        return groups, modes, q_array, \
            thetamask, tmask, \
            e_array_mask, m_array_mask, r_array_mask

    def random_f(self, alpha=None):
        """Samples f

        Samples f (15-vector) from Dirichlet(f | alpha)
        if alpha == None, an all-1 vector is used in its place

        :param alpha: Dirichlet parameter (with shape identical to f)
        :type alpha: numpy.array
        :return: a random f
        :rtype: numpy.array

        """
        expected_shape = (len(self.vfg),)
        if isinstance(alpha, type(None)):
            alpha = np.ones(expected_shape)
        else:
            assert np.array(alpha).shape == \
                expected_shape, 'alpha must have shape ' \
                                + str(expected_shape)
        return np.random.dirichlet(alpha)

    def random_theta(self, alpha=None):
        """Samples :math:`\\theta`

        Samples each row of theta (3x15-array)
        from Dirichlet(theta[c, :] | alpha[c, :])
        if alpha == None, than the following array is used
        ::
            [[1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0],
             [0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0],
             [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1]]

        :param alpha: Dirichlet parameter (with shape identical to theta)
        :type alpha: numpy.array
        :return: a random theta
        :rtype: numpy.array

        """
        expected_shape = (len(self.groups), len(self.modes))
        if isinstance(alpha, type(None)):
            alpha = self.theta_mask.copy()
        else:
            assert np.array(alpha).shape == \
                expected_shape, 'alpha must have shape ' \
                                + str(expected_shape)
            assert np.abs(np.sum(alpha * (
                np.ones_like(self.theta_mask) - self.theta_mask))) < 1e-10, \
                'alpha must be zero everywhere where theta_mask == 0'
        theta = np.zeros(expected_shape)
        for c_idx, c in enumerate(self.groups):
            # deterministically 0.0 where the input is 0.0
            theta[c_idx, :] = np.random.dirichlet(alpha[c_idx, :])
        return theta

    def random_e_array(self, alpha=None):
        """Samples E (15x3x3x3-array)

        Samples each (j,k)-block of the E (15x3x3x3-array)
        from Dirichlet(E[m,i,:,:] | alpha[m,i, :, :]
        if alpha == None, then Emask + 999 * I[i == j and i == k]
        is used for all m

        :param alpha: Dirichlet parameters (with shape identical to E)
        :type alpha: numpy.array
        :return: a random E
        :rtype: numpy.array

        """
        expected_shape = (
            len(self.modes), len(self.ig), len(self.ig), len(self.ig))
        if isinstance(alpha, type(None)):
            alpha = self.e_array_mask.copy()
            for m_idx, m in enumerate(self.modes):
                for i_idx, i in enumerate(self.ig):
                    alpha[m_idx, i_idx, i_idx, i_idx] += 999
        else:
            assert np.array(alpha).shape == \
                expected_shape, 'alpha must have shape ' \
                                + str(expected_shape)
            assert np.abs(np.sum(alpha * (np.ones_like(self.e_array_mask) -
                                          self.e_array_mask))) < 1e-10, \
                'alpha must be zero everywhere where e_array_mask == 0'
        e_array = np.zeros(expected_shape)
        for m_idx, m in enumerate(self.modes):
            for i_idx, i in enumerate(self.ig):
                alpha_mi_vec = alpha[m_idx, i_idx, :, :].flatten()
                # deterministically 0.0 where the input is 0.0
                e_array_mi_vec = np.random.dirichlet(alpha_mi_vec)
                e_array[m_idx, i_idx, :, :] = e_array_mi_vec.reshape((3, 3))
        return e_array

    def simulate_data(self, total_counts, f, theta, e_array, epsilon=1e-30):
        """Samples a random aggregate data set from total_counts, f, theta, E

        :param total_counts: total number of variants to be simulated
        :param f: fixed value for f
        :param theta: fixed value for theta
        :param e_array: fixed value for E
        :param epsilon: numerical floor for logarithm
        :return:
            * N: observable data (27x27-array of integers)
            * N_complete: hidden components (15x5x27x27-array of integers)
        :rtype: tuple

        """
        w = f
        g_counts = np.random.multinomial(total_counts, w)

        logtheta = np.log(theta + epsilon * (1.0 - self.theta_mask))
        t = self.t_array_mask * \
            np.exp(np.einsum('cm,cg->gm', logtheta, self.q_array))
        gm_counts = np.zeros((len(self.vfg), len(self.modes)))
        for g_idx, g in enumerate(self.vfg):
            gm_counts[g_idx, :] = np.random.multinomial(g_counts[g_idx],
                                                        t[g_idx, :])

        log_e_array = np.log(e_array + epsilon * (1.0 - self.e_array_mask))
        m_array = self.m_array_mask * \
            np.exp(np.einsum('mijk,ijkgGH->gmGH', log_e_array, self.k_array))
        gmg1g2_counts = np.zeros(
            (len(self.vfg), len(self.modes), len(self.fg), len(self.fg)))
        for g_idx, g in enumerate(self.vfg):
            for m_idx, m in enumerate(self.modes):
                m_array_vec = m_array[g_idx, m_idx, :, :].copy().flatten()
                gmg1g2_counts_vec = np.random.multinomial(
                    gm_counts[g_idx, m_idx], m_array_vec)
                gmg1g2_counts[g_idx, m_idx, :, :] = gmg1g2_counts_vec.reshape(
                    gmg1g2_counts.shape[2:])
        g1g2_counts = np.einsum('gmGH->GH', gmg1g2_counts)
        n_array_complete = np.einsum('gmGH->GHgm', gmg1g2_counts)
        return g1g2_counts.astype(int), n_array_complete.astype(int)

    def calculate_distribution(self, f, theta, e_array, epsilon=1e-30):
        """Returns :math:`P(G1, G2, g, m | f, \\theta, E)` (Eq. 6 in Notes.pdf)

        Returns the probability of a set of attributes (observed and hidden
        variables) given the model parameter (f, theta ,E)

        :param f: true genotype trio frequencies, of shape (15,)
        :type f: numpy.array
        :param theta: error model fractions, of shape (3, 15)
        :type theta: numpy.array
        :param e_array: error rates, of shape (15, 3, 3, 3)
        :type e_array: numpy.array
        :param epsilon: numerical floor for logarithm
        :type epsilon: float
        :return: :math:`P_{G1,G2,g,m}`, of shape (27, 27, 15, 5)
        :rtype: numpy array

        """
        w = f

        logtheta = np.log(theta + epsilon * (1.0 - self.theta_mask))
        q_array = self.q_array
        t = self.t_array_mask * \
            np.exp(np.einsum('cm,cg->gm', logtheta, q_array))

        log_e_array = np.log(e_array + epsilon * (1 - self.e_array_mask))
        k_array = self.k_array
        m_array = self.m_array_mask * \
            np.exp(np.einsum('mijk,ijkgGH->gmGH', log_e_array, k_array))

        return np.einsum('g,gm,gmGH->GHgm', w, t, m_array)
