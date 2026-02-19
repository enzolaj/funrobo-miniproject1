# FunRobo Project 1

This repository extends the template code provided for the Fundamentals of Robotics Mini-Project 1 assignment, covering the newly added implementations for the six-DOF Kinova arm and five-DOF Hiwonder arm within the visualization tool through the Denavit–Hartenberg parameters for each model. Through these parameters, we construct homogenous matrices to accurately transform the joint values for precise robotic control. Additionally, through these homogenous matrices, we are able to derive the Jacobian and Inverse-Jacobian for forward kinematics and implement resolved-rate motion control for real-time movement of the Hiwonder robot arm in response to joystick controls. For optimal functionality, we implemented the damped least squares method when calculating the Inverse-Jacobian to more effectively traverse through singularities. 


## Project Structure (Overview)

The codebase is split into two main repositories: funrobo_kinematics (math & simulation) and funrobo_hiwonder (hardware drivers & control).

```bash
funrobo_ws/
  funrobo_kinematics/             # Kinematics Library & Simulation
    funrobo_kinematics/
      core/
        arm_models.py             # Parent templates (FiveDOFRobotTemplate)
        utils.py                  # DH transforms & Matrix utilities
        visualizer.py             # PyGame visualizer
        fivedof_rrmc.py            # Part 2: 5-DOF RRMC Implementation (for importing)
    scripts/
      5dof_fk.py                  # Part 1: 5-DOF Forward Kinematics (Viz)
      6dof_fk.py                  # Part 1: 6-DOF Kinova Forward Kinematics (Viz)
      fivedof_rrmc.py             # Part 2: 5-DOF RRMC Implementation

  funrobo_hiwonder/               # Hardware Control Library
    examples/
      hiwonder_rrmc.py            # Part 3: Physical Robot Control (Gamepad)
```
*In the actual implementation, certain scripts were duplicated and moved to make importing those files easier.

## Part 1: Forward Kinematics (Simulation)

The first segment of the project covers the derivation and implementation of DH parameters for both the Hiwonder (5-DOF) and Kinova (6-DOF) arms. Implementation Details

### 5-DOF Kinematics (fivedof_rrmc.py)

This file implements the DH table for the Hiwonder robot arm and can be run with:
```bash
python funrobo_kinematics/scripts/5dof_fk.py
```

### 6-DOF Kinematics (6dof_fk.py)

This file implements the DH table for the Kinova robot arm and can be run with:
```bash
python funrobo_kinematics/scripts/6dof_fk.py
```

## Part 2: Resolved-Rate Motion Control (Simulation)

The second segment of the project covers the implementation of the Jacobian and Inverse Jacobian matrices to control the 5-DOF arm end-effector velocity in the visualizer.

### 5-DOF RRMC (fivedof_rrmc.py)

This file implements the analytical derivation of the Jacobian through the homogeneous matrices and applies damped least squares when solving for the inverse Jacobian. It can be run with:
```bash
python funrobo_kinematics/scripts/fivedof_rrmc.py
```


## Part 3: Hardware Control (Physical)

This part of the project covers the implementation of RRMC on the physical Hiwonder robot using a Logitech F310 Gamepad.

### Physical Control (hiwonder_rrmc.py)
This file implements a control loop with dead reckoning that uses the Jacobian and gamepad input to move the end effector of the Hiwonder robot in real-time. It can be run with:
```bash
python funrobo_hiwonder/examples/hiwonder_rrmc.py
```


