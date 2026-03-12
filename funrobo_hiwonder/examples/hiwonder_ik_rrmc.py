# main.py
"""
Main Application Script
----------------------------
Example code for the MP1 RRMC implementation
"""

import time
import traceback
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.fivedof_ik import FiveDOF
#from funrobo_hiwonder.examples.joystick_control import joystick_control

def main():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = FiveDOF()
        
        control_hz = 20
        dt = 1 / control_hz
        t0 = time.time()

        # Initialize target joint values
        # Maintain state for open-loop velocity integration
        print("Waiting for initial joint reading...")
        time.sleep(1)
        curr_joints_deg = robot.get_joint_values()
        if all(j == 0 for j in curr_joints_deg):
             print("[WARNING] Initial joint reading is all zeros. Check connection.")
        
        target_joints_rad = [np.deg2rad(j) for j in curr_joints_deg[:5]] 
        gripper_pos_deg = curr_joints_deg[5]
        print(f"Initial target joints (rad): {target_joints_rad}")

        # Define a list of target positions (x, y, z)
        """target_positions_shapes = [
            [-0.29, -0.06, 0.31],
            [-0.29, -0.06, 0.19],
            [-0.29, 0.06, 0.19],
            [-0.29, 0.06, 0.31],
            [-0.29, -0.06, 0.31], # Completed square
            [-0.29, 0.0, .31],
            [-0.29, -0.06, .263],
            [-0.29, -0.038, .19],
            [-0.29, 0.038, .19],
            [-0.29, 0.06, .263],
            [-0.29, 0.0, .31], # Completed Star
        ]

        target_positions = [ # Start C
            [-0.29, -0.05779, 0.27457],
            [-0.29, -0.07123, 0.27976],
            [-0.29, -0.08553, 0.27809],
            [-0.29, -0.09741, 0.26994],
            [-0.29, -0.10412, 0.2572],
            [-0.29, -0.10412, 0.2428],
            [-0.29, -0.09741, 0.23006],
            [-0.29, -0.08553, 0.22191],
            [-0.29, -0.07123, 0.22024],
            [-0.29, -0.05779, 0.22543], # Completed C
            [-0.29, -0.035, 0.212], # Start A
            [-0.29, 0.0, 0.288], 
            [-0.29, 0.035, 0.2],
            [-0.29, -0.018, 0.248], 
            [-0.29, 0.02, 0.248], # Completed A
            [-0.29, 0.04, 0.277], # Start T
            [-0.29, 0.115, 0.272],
            [-0.29, 0.0825, 0.274],
            [-0.29, 0.0825, 0.195], # Completed T
        ]"""


        initial_ee, _ = model.calc_forward_kinematics(target_joints_rad[:5])
        prev_pos = np.array([initial_ee.x, initial_ee.y, initial_ee.z])

        for target_pos in target_positions:
            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break
            
            print(f"Moving to target: {target_pos}")
            
            # Get current joints for IK initial guess
            curr_joints_deg = robot.get_joint_values()
            curr_joints_rad = [np.deg2rad(j) for j in curr_joints_deg]
            
            ee = ut.EndEffector()
            ee.x, ee.y, ee.z = target_pos[0], target_pos[1], target_pos[2]
            
            # Use only the first 5 joints (arm) for IK, pass in radians
            # ik_result_rad = model.calc_analytical_inverse_kinematics(ee, curr_joints_rad)
            ik_result_rad = model.calc_numerical_ik(ee, curr_joints_rad)
            
            # Append gripper (keep current value)
            full_joints_rad = list(ik_result_rad) + [curr_joints_rad[5]]
            
            # Move to the target position
            # Duration determines time, but not speed, so we can use an estimated optimal speed for duration
            optimal_speed = .1 #m/s
            distance = np.linalg.norm(np.array(target_pos) - np.array(prev_pos))
            move_duration = distance / optimal_speed
            robot.set_joint_values(full_joints_rad, duration=move_duration, radians=True)
            print(f"Move duration: {move_duration}")
            # Wait for the movement to complete plus a small buffer
            time.sleep(move_duration + (move_duration**2) * .5)
            prev_pos = target_pos

        print("All targets reached.")
        
        # Keep the script alive for a moment or exit
        time.sleep(1)

            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        robot.shutdown_robot()




if __name__ == "__main__":
    main()