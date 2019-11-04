__doc__ = """ Boundary conditions for rod """

import numpy as np

from ._rod import *
from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from elastica._rotations import _get_rotation_matrix

# the base class for rod boundary conditions
# also the free rod class
class FreeRod:
    def __init__(self, rod):
        self.rod = rod

    def dirichlet(self):
        pass

    def neumann(self):
        pass


# start of the rod fixed
class OneEndFixedRod(FreeRod):
    def __init__(self, rod, start_position, start_directors):
        self.rod = rod
        self.start_position = start_position
        self.start_directors = start_directors

    def dirichlet(self):
        self.rod.position[..., 0] = self.start_position
        self.rod.directors[..., 0] = self.start_directors

    def neumann(self):
        self.rod.velocity[..., 0] = 0
        self.rod.omega[..., 0] = 0

# start of the helical buckling bc
class HelicalBucklingBC(FreeRod):
    def __init__(self, rod, twisting_time, D, R, direction):
        self.rod = rod
        self.twisting_time = twisting_time
        self.D = D
        self.R = R
        self.angel_vel_scalar = (2.0 * R * np.pi / self.twisting_time) / 2.0
        self.shrink_vel_scalar = self.D / (self.twisting_time * 2.0)

        self.direction = direction

        self.final_startX = self.rod.position[...,0]  + self.D / 2.0 * self.direction
        self.final_endX   = self.rod.position[...,-1] - self.D / 2.0 * self.direction

        self.ang_vel = self.angel_vel_scalar * self.direction
        self.shrink_vel = self.shrink_vel_scalar * self.direction

        theta = R * np.pi

        self.final_startQ = _get_rotation_matrix(theta, self.direction.reshape(3,1)).reshape(3,3)\
                            @ self.rod.directors[..., 0] # rotation_matrix wants vectors 3,1
        self.final_endQ   = _get_rotation_matrix(-theta, self.direction.reshape(3,1)).reshape(3,3)\
                            @ self.rod.directors[..., -1] # rotation_matrix wants vectors 3,1

    def dirichlet(self, time):
        if time > self.twisting_time:
            self.rod.position[...,0]  = self.final_startX
            self.rod.position[...,-1] = self.final_endX

            self.rod.directors[...,0]  = self.final_startQ
            self.rod.directors[...,-1] = self.final_endQ


    def neumann(self, time):
        if time > self.twisting_time:
            self.rod.velocity[...,0]  = 0.0
            self.rod.omega[...,0]     = 0.0

            self.rod.velocity[...,-1] = 0.0
            self.rod.omega[...,-1]    = 0.0

        else:
            self.rod.velocity[...,0]  =  self.shrink_vel
            self.rod.omega[...,0]     =  self.ang_vel

            self.rod.velocity[...,-1]  = -self.shrink_vel
            self.rod.velocity[...,-1]  = -self.ang_vel








