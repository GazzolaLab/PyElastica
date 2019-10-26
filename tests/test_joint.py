__doc__ = """ Joint between rods test module """

# System imports
import numpy as np
from elastica.joint import FreeJoint
from numpy.testing import assert_allclose, assert_array_equal
from elastica.utils import Tolerance

def test_freejoint():
     # Define the rod for testing
     class rod:
         def __init__(self, n,r,v):
             self.position = r
             self.velocity = np.tile(v, (n+1,1))
             self.external_forces = np.tile(np.array([0.0,0.0,0.0]), (n+1,1))


     # Origin of the rod
     origin1 = np.array([0.0,0.0,0.0])
     origin2 = np.array([1.1,0.0,0.0])

     # Number of elements
     n = 2

     # Rod positions
     L=1
     x = np.linspace(0.,L,n+1)
     r1 = np.transpose(np.array([[i,0.,0.] for i in x]) + origin1)
     r2 = np.transpose(np.array([[i,0.,0.] for i in x]) + origin2)

     # Rod velocity
     v1 = np.array([-1,0,0])
     v2 = v1*-1

     # Create two rod classes
     rod1 = rod(n,r1,v1)
     rod2 = rod(n,r2,v2)

     # Stiffness between points
     k = 1e8

     # Damping between two points
     nu = 1

     # Rod indexes
     rod1_index = -1
     rod2_index = 0

     # Compute the free joint forces
     distance = rod2.position[...,rod2_index]-rod1.position[...,rod1_index]
     elasticforce = k * distance
     relative_vel = rod2.velocity[...,rod2_index] - rod1.velocity[...,rod1_index]
     normal_relative_vel = np.dot(relative_vel,distance)/np.linalg.norm(distance)
     dampingforce = nu * normal_relative_vel * distance
     contactforce = elasticforce - dampingforce


     frjt = FreeJoint(k,nu,rod1,rod2,rod1_index,rod2_index)

     frjt.apply_force()

     assert_allclose(frjt.rod_one.external_forces[...,rod1_index],contactforce, atol=Tolerance.atol())
     assert_allclose(frjt.rod_two.external_forces[...,rod2_index],-1*contactforce, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])




