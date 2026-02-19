from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOF(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    # Since we use the homogenous matrices in the jacobian calculation, we define 
    # two helper functions to process the joint values and compute the transformation matrices
    # and put them in both functions. These functions use  the example code from Kene.
    def _process_joints(self, joint_values, radians=True):
        """
        Helper to handle unit conversion and joint limit clipping.
        """
        curr_joint_values = np.array(joint_values, dtype=float)

        if not radians:
            curr_joint_values = np.deg2rad(curr_joint_values)

        # Clip joint angles to limits
        # Transpose limits to get min_limits and max_limits arrays
        limits = np.array(self.joint_limits)
        curr_joint_values = np.clip(curr_joint_values, limits[:, 0], limits[:, 1])
        
        return curr_joint_values

    def _compute_transforms(self, joint_values):
        """
        Helper to calculate cumulative transformation matrices (H_cumulative)
        and individual joint transforms (Hlist) based on the FiveDOF robot model.
        
        Args:
            joint_values (list/array): Processed joint angles in radians.
        """
        theta = joint_values
        DH = np.array([ # DH parameters for each joint
            [theta[0],          self.l1,         0,       -pi/2],
            [theta[1]-(pi/2),   0,               self.l2,  pi],
            [theta[2],          0,               self.l3,  pi],
            [theta[3]+(pi/2),   0,               0,        pi/2],
            [theta[4],          self.l4+self.l5, 0,        0]
        ])

        Hlist = [ut.dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

        # Compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK).
        """
        q = self._process_joints(joint_values, radians) # Get joint values after clipping
        
        H_cumulative, Hlist = self._compute_transforms(q) # Compute homogenous matrices, get transform from joint 1 to EE
        H_ee = H_cumulative[-1] # Get transform from joint 1 to EE

        ee = ut.EndEffector() # Initialize end effector
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3] # Get position of EE
        rpy = ut.rotm_to_euler(H_ee[:3, :3]) # Get RPY of EE
        ee.rotx, ee.roty, ee.rotz = rpy # Set RPY of EE

        return ee, Hlist # Return end effector and transform from joint 1 to EE

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy, vz].
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate joint velocities using the inverse Jacobian
        vel = vel[:3]  # Consider only the first three components of the velocity
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel
        
        joint_vel = np.clip(joint_vel, 
                            [limit[0] for limit in self.joint_vel_limits], 
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        return new_joint_values

    def jacobian(self, joint_values: list, radians=True):
        """
        Returns the Geometric Jacobian matrix for the robot.
        """
        q = self._process_joints(joint_values, radians) # Get joint values after clipping
        H_cumulative, _ = self._compute_transforms(q) # Compute homogenous matrices, get transform from joint 1 to EE
        p_ee = H_cumulative[-1][:3, 3] # Get position of EE
        J = np.zeros((6, self.num_dof)) # Initialize Jacobian matrix

        for i in range(self.num_dof): # Compute Jacobian for each joint
            transform_i = H_cumulative[i] # Get transform from joint i to EE

            z_axis = transform_i[:3, 2] # Get z-axis of joint i
            p_joint = transform_i[:3, 3] # Get position of joint i

            J[:3, i] = np.cross(z_axis, (p_ee - p_joint)) # Linear velocity component (cross product of z-axis and distance to EE)
            J[3:, i] = z_axis # Angular velocity component (z-axis)
            
        return J[:3, :] # Only account for translation speed
    
    def inverse_jacobian(self, joint_values: list, radians=True):
        """
        Returns the inverse of the Jacobian matrix.

        Args:
            joint_values (list): Joint angles.
            radians (bool): Whether joint_values are in radians (True) or degrees (False).

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        jacobian = self.jacobian(joint_values, radians=radians)
        lamda = 0.0001
        return jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T + lamda * np.eye(jacobian.shape[0]))





if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()