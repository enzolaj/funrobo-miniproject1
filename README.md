# FunRobo Mini-Projects 1 & 2

This repository contains the implementations for the Fundamentals of Robotics Mini-Projects 1 and 2. It covers Forward Kinematics (FK), Resolved-Rate Motion Control (RRMC), Inverse Kinematics (IK), and motion planning for the Kinova (6-DOF) and Hiwonder (5-DOF) robot arms.

## Project Structure (Overview)

The codebase is split into two main repositories: `funrobo_kinematics` (math & simulation) and `funrobo_hiwonder` (hardware drivers & control).

```bash
funrobo_ws/
  funrobo_kinematics/             # Kinematics Library & Simulation
    funrobo_kinematics/
      core/
        arm_models.py             # Parent templates (FiveDOFRobotTemplate)
        fivedof.py                # 5-DOF Model with FK & IK implementations
        kinova.py                 # 6-DOF Model with FK & IK implementations
        utils.py                  # DH transforms & Matrix utilities
        visualizer.py             # PyGame visualizer

      
  funrobo_hiwonder/               # Hardware Control Library
    examples/
      hiwonder_ik_rrmc.py         # Part 4: IK & Shape Tracing
```

## Part 1: Forward Kinematics (Simulation)

The first segment covers the derivation and implementation of DH parameters for both the Hiwonder (5-DOF) and Kinova (6-DOF) arms.

### 5-DOF Kinematics (fivedof.py)
Implements the DH table for the Hiwonder robot arm.
```bash
python funrobo_kinematics/funrobo_kinematics/core//fivedof.py
```

### 6-DOF Kinematics (fivedof.py)
Implements the DH table for the Kinova robot arm.
```bash
python funrobo_kinematics/funrobo_kinematics/core/kinova.py
```

## Part 2: Resolved-Rate Motion Control (Simulation)

The second segment covers the implementation of the Jacobian and Inverse Jacobian matrices to control the 5-DOF arm end-effector velocity in the visualizer.

### 5-DOF RRMC (fivedof.py)
Implements analytical Jacobian derivation and Damped Least Squares (DLS) for the inverse Jacobian.
```bash
python funrobo_kinematics/funrobo_kinematics/core//fivedof.py
```

## Part 3: Hardware Control (Physical)

This part covers the implementation of RRMC on the physical Hiwonder robot using a Logitech F310 Gamepad.

### Physical Control (hiwonder_rrmc.py)
Implements a control loop with dead reckoning that uses the Jacobian and gamepad input to move the end effector in real-time.
```bash
python funrobo_hiwonder/examples/hiwonder_rrmc.py
```

## Part 4: Inverse Kinematics & Path Planning

This section covers the implementation of Inverse Kinematics (IK) to allow the 5-DOF arm to reach specific Cartesian coordinates. We implemented both numerical and analytical IK solvers.

### IK Implementation (fivedof.py)
The `FiveDOF` class in `funrobo_kinematics/core/fivedof.py` now includes:
- `calc_numerical_ik`: Solves for joint angles given a target end-effector pose using an iterative Jacobian pseudo-inverse method with joint limit enforcement.
- `calc_inverse_kinematics`: An analytical approach using kinematic decoupling.

### Shape Tracing (hiwonder_ik_rrmc.py)
This script utilizes the IK solver to make the physical robot trace defined shapes (Square, Star, letters C-A-T) in 3D space.
```bash
python funrobo_hiwonder/examples/hiwonder_ik_rrmc.py
```

