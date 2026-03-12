
from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import KinovaRobotTemplate

class Kinova(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()
        # Override joint limits if needed, or use defaults from template
        # self.joint_limits = ... 

    def _compute_transforms(self, joint_values):
        """
        Helper to calculate cumulative transformation matrices (H_cumulative)
        and individual joint transforms (Hlist) based on the Kinova robot model.
        
        Args:
            joint_values (list/array): Processed joint angles in radians.
        """
        theta = joint_values
        
        # DH parameters
        # Note: The Kinova model uses 7 frames for 6 joints (Frame 0 is base offset)
        DH = np.array([
            [0,                 0,            0,          pi],
            [theta[0],          -self.l2-self.l1,           0,          pi/2],
            [theta[1]-(pi/2),   0,                  self.l3,   pi],
            [theta[2]-(pi/2),   0,                  0,          pi/2],
            [theta[3],          -self.l4-self.l5,   0,          -pi/2],
            [theta[4],          0,                  0,          pi/2],
            [theta[5],          -self.l6-self.l7,   0,          pi]
        ])

        Hlist = [ut.dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

        # Compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        curr_joint_values = joint_values.copy()

        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
        
        H_cumulative, Hlist = self._compute_transforms(curr_joint_values)

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist

    def calc_numerical_ik(self, ee, joint_values, tol=0.002, ilimit=100):
        """
        Numerical IK with angles wrapped to [-pi, pi] and joint limit enforcement.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles (radians).
            tol (float, optional): Convergence tolerance. Defaults to 0.002.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians, wrapped to [-pi, pi] and within joint limits.
        """
        # unpack end effector position and rotation
        x_target, y_target, z_target = ee.x, ee.y, ee.z
        new_joint_values = np.array(joint_values, dtype=float)

        for _ in range(100):  # allow 100 attempted starting configurations
            for _ in range(ilimit): # 100 iterations for each attempted configuration
                # get the end effector position based on the current joint guess and find the error from the desired position
                current_ee, _ = self.calc_forward_kinematics(new_joint_values)
                error = np.array([x_target, y_target, z_target]) - np.array([current_ee.x, current_ee.y, current_ee.z])

                # if the error is within tolerance, return the joint angle solution
                if np.linalg.norm(error) <= tol:
                    return new_joint_values

                # get next iteration by updating with inverse jacobian
                new_joint_values += self.inverse_jacobian(new_joint_values) @ error

                # enforce joint limits
                for i, (low, high) in enumerate(self.joint_limits):
                    new_joint_values[i] = np.clip(new_joint_values[i], low, high)

            # if not converged, return a random configuration and try again
            new_joint_values = np.array(ut.sample_valid_joints(self), dtype=float)
        
        # return null if not converged
        return np.zeros(len(joint_values))

    def calc_inverse_kinematics(self, ee, joint_values=None, soln=0):
        """
        Calculates the analytical inverse position kinematics for the 6-DOF Kinova manipulator.
        Utilizes kinematic decoupling via the spherical wrist to compute the closed-form 
        solution governed by the explicitly requested geometric root.
        """
        p_ee = np.array([ee.x, ee.y, ee.z])
        R_06 = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
        d_6 = self.l6 + self.l7
        
        # Wrist Position
        p_wrist = p_ee - d_6 * (R_06 @ np.array([0, 0, 1]))
        wx, wy, wz = p_wrist[0], p_wrist[1], p_wrist[2]

        solutions = []

        theta1_configs = [
            (-atan2(wy, wx), np.sqrt(wx**2 + wy**2)),
            (-atan2(-wy, -wx), -np.sqrt(wx**2 + wy**2))
        ]

        for theta_1, r in theta1_configs:
            s = wz - (self.l1 + self.l2) # Height relative to shoulder
            
            # Law of Cosines for Elbow (beta) and Shoulder (alpha)
            l_proximal = self.l3
            l_distal = self.l4 + self.l5
            L_sq = r**2 + s**2
            
            # Check reachability
            numerator = l_proximal**2 + l_distal**2 - L_sq
            denominator = 2 * l_proximal * l_distal
            if abs(numerator) > abs(denominator):
                continue

            cos_beta = numerator / denominator
            beta = np.arccos(cos_beta)

            # 2. Elbow Config (Up vs Down)
            for theta_3 in [np.pi - beta, -(np.pi - beta)]:
                
                # alpha: Interior angular offset 
                alpha = np.arctan2(l_distal * np.sin(theta_3), l_proximal + l_distal * np.cos(theta_3))

                # gamma: Absolute angular trajectory
                gamma = np.arctan2(s, r)

                # theta_2
                theta_2 = (np.pi / 2) - (gamma - alpha)
                
                # Calculate Rotation of Frame 3 (R_03)
                q_temp = [theta_1, theta_2, theta_3, 0, 0, 0]
                H_cumulative, _ = self._compute_transforms(q_temp)
                R_03 = H_cumulative[4][:3, :3]
                
                R_36 = R_03.T @ R_06
                
                # Wrist Config (Positive vs Negative sine for theta_5)
                c5 = -R_36[2, 2]
                s5_mag = np.sqrt(np.clip(1 - c5**2, 0, 1))
                
                for s5 in [s5_mag, -s5_mag]:
                    theta_5 = atan2(s5, c5)
                    
                    if abs(s5) > 1e-6:
                        theta_4 = atan2(-R_36[1, 2], -R_36[0, 2])
                        theta_6 = atan2(-R_36[2, 1], -R_36[2, 0])
                    else:
                        # Gimbal lock
                        theta_4 = 0
                        theta_6 = atan2(R_36[1, 0], R_36[0, 0])

                    candidate_q = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
                    candidate_q = [self.normalize_angle(q) for q in candidate_q]

                    # Check limits
                    if ut.check_joint_limits(candidate_q, self.joint_limits):
                        solutions.append(candidate_q)

        # Sort solutions by error
        def get_error(q):
            fk_ee, _ = self.calc_forward_kinematics(q)
            return np.linalg.norm(np.array([ee.x, ee.y, ee.z]) - np.array([fk_ee.x, fk_ee.y, fk_ee.z]))
        
        solutions.sort(key=get_error)

        if not solutions:
            return [0, 0, 0, 0, 0, 0]

        # If a specific solution index is requested, return it
        if soln < len(solutions):
            return solutions[soln]
        
        # Return best solution
        return solutions[0]

    def normalize_angle(self, angle):
        """Normalize an angle to the range (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot. 

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (6x6).
        """
        
        curr_joint_values = joint_values.copy()

        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
        
        H_cumulative, _ = self._compute_transforms(curr_joint_values)

        p_ee = H_cumulative[-1][:3, 3] 
        J = np.zeros((6, self.num_dof)) 
        
        for i in range(self.num_dof):
            
            transform = H_cumulative[i+1]
            z_axis = transform[:3, 2] # Z-axis of the frame about which theta[i] rotates?
            
            transform_axis = H_cumulative[i+1] # Frame defining the Z axis
            
            z_axis = transform_axis[:3, 2]
            p_joint = transform_axis[:3, 3]
            
            J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
            J[3:, i] = z_axis
            
        return J

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))

if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim

    model = Kinova()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
