import random, math
import pytest
import yaml

import funrobo_kinematics.core.utils as ut

#TODO Import your robot model script
# ---------------------------------------
from examples.kinova import Kinova
# ---------------------------------------


robot_model = Kinova()
N = 100 # number of sample tries


# -----------------------------------------------------------------------------
# Test description
# -----------------------------------------------------------------------------
# - This script uses pytest to run unit tests on the inverse and forward kinematics 
#   functions we implement.
#
# - STEPS FOR INVERSE KINEMATICS:
#   1. Make sure you import the right robot model class into the script
#   2. The script randomly generates N number of valid joint position and end-effector pose pairs
#   3. For each pair, it computes the joint values using the IK given the end-effector pose
#   4. The test checks if the computed solution is valid


joint_values_list = [ut.sample_valid_joints(robot_model) for _ in range(N)]
ee_list = []

for joint_values in joint_values_list:
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)
    ee_list.append([float(ee.x), float(ee.y)])

ids = [f"joint_values_{i}={[round(x,2) for x in q]}" for i, q in enumerate(joint_values_list)]


# -----------------------------------------------------------------------------
# Python test for analytical inverse kinematics
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("joint_values", joint_values_list, ids=ids)
def test_analytical_ik(joint_values):
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

    init_joint_values = ut.sample_valid_joints(robot_model)
    new_joint_values = robot_model.calc_inverse_kinematics(ee, init_joint_values, soln=0)

    assert ut.check_valid_ik_soln(new_joint_values, ee, robot_model)