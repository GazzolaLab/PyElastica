#!/usr/bin/env python3

__author__ = "Fan Kiat Chan"
__copyright__ = "Copyright 2019, Elastica Python"
__credits__ = ["Fan Kiat Chan"]
__license__ = "MPL 2.0"
__version__ = "0.1.0"
__maintainer__ = "Fan Kiat Chan"
__email__ = "fchan5@illinois.edu"
__status__ = "Production"


class positionVerlet2nd(object):
    # In the case of no boundary conditions applied
    def pass_dirichlet(*args, **kwargs):
        pass

    def pass_neumann(some_obj, rhs_linear, rhs_angular, *args, **kwargs):
        return rhs_linear, rhs_angular

    def __init__(
        self, obj, dirichletBC=pass_dirichlet, neumannBC=pass_neumann, *args, **kwargs
    ):
        self.obj = obj
        self.dt_prev = 1.0
        self.dirichletBC = dirichletBC
        self.neumannBC = neumannBC
        self.time = 0.0

        self.__dict__.update(kwargs)

    def step(self, dt):
        """

        === Phase 1 ===
        (1) r(t + dt/2) = r(t) + dt/2 * v(t)
        (2) Q(t + dt/2) = exp(dt/2 w(t)) * Q(t)
        ===============

        (3) Apply Dirichlet boundary conditions

        === Phase 2 ===
        (4) v(t + dt) = v(t) + dt * dv/dt(t + dt/2)
            -> compute dv/dt
            -> compute contact forces
            -> update dv/dt += contact forces / mass (Neumann BC)
            -> compute v(t+dt)
        (5) w(t + dt) = w(t) + dt * dw/dt(t + dt/2)
            -> compute dw/dt
            -> compute associated torque from contact
            -> update dwdt += associated torque / (J/e) (Neumann BC)
            -> compute w(t+dt)
        ================

        (6) Apply Dirichlet boundary conditions

        === Phase 3 ===
        (7) r(t + dt) = r(t + dt/2) + dt/2 * v(t + dt)
        (8) Q(t + dt) = exp(dt/2 w(t + dt)) * Q(t + dt/2)
        ===============
        """

        """################### Phase 1 ######################"""
        self.obj.r += dt / 2.0 * self.obj.v
        rot = self.obj.rotate_v(dt / 2.0, self.obj.w)
        self.obj.Q = rot @ self.obj.Q
        ###################################################

        # Apply BC and update current time
        self.dirichletBC(self.obj, self.time)
        self.time += dt / 2.0

        """#################### Phase 2 ######################"""
        """##### Linear and angular velocity"""
        rhs_linear = self.obj.compute_rhs_linearmom()
        rhs_angular = self.obj.compute_rhs_angularmom(self.dt_prev)

        # forces and torque goes here (ie contact, friction, muscular activity)
        rhs_linear, rhs_angular = self.neumannBC(
            self.obj, rhs_linear, rhs_angular, self.time
        )

        self.obj.v += dt * rhs_linear
        self.obj.w += dt * rhs_angular
        ###################################################

        # Apply BC
        self.dirichletBC(self.obj, self.time)

        """#################### Phase 3 ######################"""
        self.obj.r += dt / 2.0 * self.obj.v
        rot = self.obj.rotate_v(dt / 2.0, self.obj.w)
        self.obj.Q = rot @ self.obj.Q
        ###################################################

        # Store previous time step and update current time
        self.dt_prev = dt
        self.time += dt / 2.0
