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
from funrobo_kinematics.core.fivedof_rrmc import FiveDOF

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
        print(f"Initial target joints (rad): {target_joints_rad}")

        while True:
            t_start = time.time()

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break

            if robot.gamepad.cmdlist:
                cmd = robot.gamepad.cmdlist[-1]


                if cmd.arm_home:
                    print("Moving Home...")
                    robot.move_to_home_position()
                    continue 

                curr_joints_deg = robot.get_joint_values()
                
                speed = 2.0  # Speed multiplier since damping slows
                vel = [cmd.arm_vx * speed, cmd.arm_vy * speed, cmd.arm_vz * speed]

                # Integrate velocity to update target position
                target_joints_rad = model.calc_velocity_kinematics(target_joints_rad, vel, dt=dt) # dt should match control freq.

                new_joints_rad = target_joints_rad

                new_joints_deg = [np.rad2deg(j) for j in new_joints_rad]
                full_joints_deg = new_joints_deg + [curr_joints_deg[5]] # The velocities are for 5 joints, but the model has a 6th (gripper)
                
                # Check if we are hitting limits
                for i, (val, limit) in enumerate(zip(new_joints_rad, model.joint_limits)):
                    if val <= limit[0] + 0.01 or val >= limit[1] - 0.01:
                        print(f"[WARNING] Joint {i+1} hitting limit: {val:.2f} (Limits: {limit[0]:.2f}, {limit[1]:.2f})")

                robot.set_joint_values(full_joints_deg, duration=dt, radians=False)

            elapsed = time.time() - t_start
            remaining_time = dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)

            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        robot.shutdown_robot()




if __name__ == "__main__":
    main()


