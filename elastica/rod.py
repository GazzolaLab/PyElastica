#!/usr/bin/env python3

import numpy as np

__author__ = "Fan Kiat Chan"
__copyright__ = "Copyright 2019, Elastica Python"
__credits__ = ["Fan Kiat Chan"]
__license__ = "MPL 2.0"
__version__ = "0.1.0"
__maintainer__ = "Fan Kiat Chan"
__email__ = "fchan5@illinois.edu"
__status__ = "Production"


class Tolerance:
    @staticmethod
    def tol():
        return np.finfo(float).eps * 10.0


class rod(object):
    ###########################
    # n elements, n+1 nodes
    ###########################

    # things to put as kwargs: v, t, shearStrain0 (intrinsic shearStrain),
    # k0 (k_intrinsic)
    def __init__(self, n, r, rho, yng_mod, shr_mod, gamma=0.0, *args, **kwargs):

        # Some constants
        self.rho = rho
        self.yng_mod = yng_mod  # young's modulus
        self.shr_mod = shr_mod  # shear modulus
        self.alpha_c = 4.0 / 3.0
        self.gamma = gamma  # dissipation
        self.gravity = False

        ###########################
        # (n+1)-sized stuff (everything consistent with size of r)
        ###########################
        self.n = n
        self.r = r

        self.v = np.zeros((n + 1, 3))
        if kwargs.__contains__("v"):
            self.v = kwargs["v"]
        assert self.v.shape == self.r.shape

        self.f_ext = np.zeros((n + 1, 3))
        if kwargs.__contains__("f_ext"):
            self.f_ext = np.array(kwargs["f_ext"])
        assert self.f_ext.shape == self.r.shape

        ###########################
        # (n)-sized stuff (everything consistent with size of l)
        ###########################
        self.l_org = np.linalg.norm(self.compute_l(), axis=1)
        self.l = self.compute_l()
        self.e = self.compute_e()
        self.e_prev = self.e
        self.tangent = self.compute_tangent()

        self.c_ext = np.zeros((n, 3))
        if kwargs.__contains__("c_ext"):
            self.c_ext = np.array(kwargs["c_ext"])
        assert self.c_ext.shape == self.l.shape

        """ DEFAULT INITIALIZATION OF DIRECTORS
        ### assumes rod lives on 2d plane, with z-axis as the normal to wall
        ### (out of plane)
        ### hence assumes d3 lives only on xy-plane (like a snake) """
        self.d3 = self.tangent
        self.d1 = np.array([[0.0, 0.0, 1.0] for i in range(self.n)])
        d2 = np.cross(self.d1, self.d3)
        self.d2 = d2 / np.linalg.norm(d2, axis=1)[:, None]
        if kwargs.__contains__("d1"):
            if kwargs.__contains__("d2") and kwargs.__contains__("d3"):
                self.d1 = np.array(kwargs["d1"])
                assert self.d1.shape == self.l.shape
                self.d2 = np.array(kwargs["d2"])
                assert self.d2.shape == self.l.shape
                self.d3 = np.array(kwargs["d3"])
                assert self.d3.shape == self.l.shape
            else:
                raise ValueError("All d1, d2, d3 has to be provided!")
        # construct Q after getting the directors
        self.Q = self.compute_Q()
        for i in range(self.n):
            assert np.all(
                np.isclose(self.Q[i].dot(self.Q[i].T) - np.eye(3), 0.0)
            ), "Q constructed is not orthonormal!"

        self.shearStrain = self.compute_shearStrain()

        self.w = np.zeros((n, 3))
        if kwargs.__contains__("w"):
            self.w = np.array(kwargs["w"])
        assert self.w.shape == self.l.shape

        self.radii = np.ones(n) * np.sqrt(1.0 / np.pi)
        if kwargs.__contains__("radii"):
            self.radii = np.array(kwargs["radii"])
        assert self.radii.shape == self.l_org.shape
        self.A_org = np.pi * self.radii ** 2

        # this is technically (n+1)-sized, but we need area defined first
        self.mass = self.compute_mass()
        assert self.mass.shape == (self.n + 1,)

        # let's store them (I, J, B, S) as a vector since they are diagonal
        self.I_org = np.array(
            [
                self.A_org[i] ** 2 / (4.0 * np.pi) * np.array([1, 1, 2])
                for i in range(self.n)
            ]
        )

        self.J_org = self.rho * (self.I_org * self.l_org[:, None])
        self.B_org = self.I_org * np.array([self.yng_mod, self.yng_mod, self.shr_mod])
        self.S_org = self.A_org[:, None] * np.array(
            [self.shr_mod * self.alpha_c, self.shr_mod * self.alpha_c, self.yng_mod]
        )

        ###########################
        # (n-1)-sized stuff
        ###########################
        self.D_org = self.compute_vorn_domain()
        self.D = self.compute_vorn_domain()
        self.e_vorn = self.compute_e_vorn()
        self.k = self.inverse_rotate_v()
        self.k_org = self.inverse_rotate_v()
        self.B_vorn_org = self.compute_B_vorn()

        ###########################
        # Other stuff that we want to add later
        # i.e. snake bspline, etc.
        ###########################
        self.__dict__.update(kwargs)

    def compute_l(self,):
        return self.r[1:] - self.r[0:-1]

    def compute_e(self,):
        return np.linalg.norm(self.r[1:] - self.r[0:-1], axis=1) / self.l_org

    def compute_tangent(self,):
        # SLOW
        # return np.array(
        #     [self.l[i] / np.linalg.norm(self.l[i]) for i in range(self.n)])
        return self.l / np.linalg.norm(self.l, axis=1)[:, None]

    def compute_Q(self,):
        return np.array([self.d1[:], self.d2[:], self.d3[:]]).transpose(1, 0, 2)

    def compute_shearStrain(self):
        # TOO SLOW
        # return np.array(
        #    [np.dot(self.Q[i], (self.tangent[i] * self.e[i] - self.Q[i,2]) )
        #    for i in range(self.n) ])
        t = self.tangent * self.e[:, None] - self.Q[:, 2]
        return (self.Q @ t[:, :, None]).transpose(0, 2, 1)[:, 0, :]

    def compute_mass(self,):
        m = self.rho * (self.A_org / self.e) * self.l_org
        mass = np.zeros(self.r.shape[0])
        mass = np.pad(m, (0, 1), "constant")
        mass = (mass + np.roll(mass, 1)) / 2.0
        return mass

    def compute_vorn_domain(self,):
        length = np.linalg.norm(self.l, axis=1)
        return (length[1:] + length[0:-1]) / 2.0

    def compute_e_vorn(self,):
        return self.D / self.D_org

    def compute_B_vorn(self,):
        Biplus1 = self.B_org[1:] * self.l_org[1:][:, None]
        Bi = self.B_org[0:-1] * self.l_org[0:-1][:, None]
        return (Biplus1 + Bi) / (2.0 * self.D_org[:, None])

    def inverse_rotate_v(self,):
        """
        Essential the inverse rotation using Rodrigues,
        but we do it for every element here
        k = log(Q[i+1] @ Q[i]) / D_org[i]
        """
        # transpose tuple (0,2,1) properly tranposes each rod element's matrix
        R = self.Q[1:] @ self.Q[:-1].transpose(0, 2, 1)
        theta = np.around((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, 12)
        theta = np.arccos(theta)

        far = np.invert(np.isclose(theta, 0.0, atol=Tolerance.tol()))

        k = np.zeros((self.n - 1, 3))
        theta_u = np.empty((self.n - 1, 3, 3))

        if np.any(far):
            theta_u[far] = (
                theta[far, None, None] / (2.0 * np.sin(theta[far, None, None]))
            ) * (R[far] - R.transpose(0, 2, 1)[far])
            k[far, 0] = theta_u[far, 2, 1] / self.D_org[far]
            k[far, 1] = theta_u[far, 0, -1] / self.D_org[far]
            k[far, 2] = theta_u[far, 1, 0] / self.D_org[far]

        return k

    # Slower version, use inverse_rotate_v
    def inverse_rotate(self,):
        k = np.zeros((self.n - 1, 3))
        for i in range(self.n - 1):
            R = np.dot(self.Q[i + 1], self.Q[i].T)
            t = (np.trace(R) - 1.0) / 2.0
            t = np.around(t, 12)
            theta = np.arccos(t)
            if np.isclose(theta, 0.0, atol=Tolerance.tol()):
                k[i] = np.array([0.0, 0.0, 0.0])
            else:
                theta_u = (R - R.T) * theta / (2.0 * np.sin(theta))
                k[i] = (
                    np.array([theta_u[2, 1], theta_u[0, -1], theta_u[1, 0]])
                    / self.D_org[i]
                )
        return k

    # slower version, use rotate_v
    def rotate(self, theta, u):
        def normalize(v):
            """ Normalize a vector/ matrix """
            norm = np.linalg.norm(v)
            if np.isclose(norm, 0.0, atol=Tolerance.tol()):
                # raise RuntimeError(
                #     "Not rotating because axis specified to be zero")
                # return v
                return np.zeros(3)
            return v / norm

        def skew_symmetrize(v):
            """ Generate an orthogonal matrix from vector elements"""
            skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
            return skv - skv.T

        rot_matrix = np.zeros((self.n, 3, 3))
        theta *= np.linalg.norm(u, axis=1)
        for i in range(self.n):
            # Convert about to np.array and normalize it
            unorm = normalize(np.array(u[i]))

            # Form the 2D Euler rotation matrix
            c_angle = np.cos(theta[i])
            s_angle = np.sin(theta[i])

            # DS for 3D Euler rotation matrix
            # Composed of 2D matrices
            tmp = np.eye(3)
            U_mat = skew_symmetrize(unorm)
            rot_matrix[i] = tmp + U_mat @ (s_angle * tmp + (1.0 - c_angle) * U_mat)

        return rot_matrix

    def rotate_v(self, theta, u):
        '''
        def normalize(v):
            """ Normalize a vector/ matrix """
            norm = np.linalg.norm(v)
            if np.isclose(norm, 0.0, atol = Tolerance.tol()):
                return np.zeros(3)
            return v / norm
        '''

        def skew_symmetrize(v):
            """ Generate an orthogonal matrix from vector elements"""
            skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
            return skv - skv.T

        rot_matrix = np.array([np.eye(3) for i in range(self.n)])

        mag = np.linalg.norm(u, axis=1)
        theta *= mag

        rot_idx = np.invert(np.isclose(mag, 0.0, atol=Tolerance.tol()))

        unorm = np.zeros_like(u)
        if np.any(rot_idx):
            unorm[rot_idx] = u[rot_idx] / mag[rot_idx, None]

            c_angle = np.cos(theta)
            s_angle = np.sin(theta)

            U_mat = np.array([skew_symmetrize(uu) for uu in unorm])
            nrot_idx = np.sum(rot_idx)
            tmp_eye = np.array([np.eye(3) for i in range(nrot_idx)])
            rot_matrix[rot_idx] = tmp_eye + U_mat[rot_idx] @ (
                s_angle[rot_idx, None, None] * tmp_eye
                + (1.0 - c_angle[rot_idx, None, None]) * U_mat[rot_idx]
            )

        return rot_matrix

    def compute_rhs_linearmom(self,):
        self.updateState()
        # TOO SLOW
        # tmp = np.array([
        #             np.dot(
        #                 np.dot(self.Q[i].T, np.diag(
        #                     self.S_org[i]/self.e[i])), self.shearStrain[i])
        #             for i in range(self.n)])
        tmp = (
            (self.Q.transpose(0, 2, 1) * (self.S_org / self.e[:, None])[:, None])
            @ self.shearStrain[:, :, None]
        ).transpose(0, 2, 1)[:, 0, :]
        fv = -self.gamma * self.v
        grav = self.gravity * (-9.81 * np.array([0.0, 0.0, 1.0]))
        dvdt = (self.delta_h_op(tmp) + self.f_ext + fv) / self.mass[:, None] + grav

        return dvdt

    def compute_rhs_angularmom(self, dt_prev):
        # TOO SLOW
        ##############################
        # bendtwist1 = np.array([
        #             np.diag(self.B_vorn_org[i]/self.e_vorn[i]**3) @ self.k[i]
        #             for i in range(self.n-1)])
        #
        # bendtwist2 = np.array([
        #             np.cross(self.k[i], np.diag(
        #                 self.B_vorn_org[i]/self.e_vorn[i]**3) @ self.k[i]) *
        #             self.D_org[i] for i in range(self.n-1)])
        #
        shearstretch = np.array(
            [
                -np.cross(
                    np.dot(self.Q[i], self.tangent[i]),
                    np.dot(np.diag(self.S_org[i]), self.shearStrain[i]),
                )
                * self.l_org[i]
                for i in range(self.n)
            ]
        )

        bendtwist1 = self.B_vorn_org * self.k / self.e_vorn[:, None] ** 3
        bendtwist2 = (
            np.cross(self.k, self.B_vorn_org / self.e_vorn[:, None] ** 3 * self.k)
            * self.D_org[:, None]
        )

        tmp1 = (self.Q @ self.tangent[:, :, None]).transpose(0, 2, 1)[:, 0, :]
        tmp2 = self.S_org * self.shearStrain
        shearstretch = -np.cross(tmp1, tmp2) * self.l_org[:, None]

        lagrg_transport = np.cross(self.J_org * self.w / self.e[:, None], self.w)
        unsteady_dilatation = (
            self.J_org
            * self.w
            / self.e[:, None] ** 2
            * (self.e - self.e_prev)[:, None]
            / dt_prev
        )

        cv = -self.gamma * self.w * self.e[:, None] * self.l_org[:, None]

        rhs = (
            self.delta_h_op(bendtwist1)
            + self.A_h_op(bendtwist2)
            + shearstretch
            + lagrg_transport
            + unsteady_dilatation
            + self.c_ext
            + cv
        )
        dwdt = rhs * self.e[:, None] / self.J_org

        return dwdt

    def delta_h_op(self, v):
        v = np.pad(v, ((0, 1), (0, 0)), "constant")
        return v - np.roll(v, 1, axis=0)

    def A_h_op(self, v):
        v = np.pad(v, ((0, 1), (0, 0)), "constant")
        return (v + np.roll(v, 1, axis=0)) / 2.0

    def updateState(self,):
        self.e_prev = self.e
        self.l = self.compute_l()
        self.e = self.compute_e()
        self.D = self.compute_vorn_domain()
        self.e_vorn = self.compute_e_vorn()
        self.tangent = self.compute_tangent()
        self.shearStrain = self.compute_shearStrain()
        self.k = self.inverse_rotate_v()
