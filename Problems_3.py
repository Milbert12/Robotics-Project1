import numpy as np
import mujoco as mj
import mujoco.viewer
import os
from math import pi

xml_path = "Furuta_Model.xml"
dirname = os.path.dirname(__file__)
abs_path = os.path.join(dirname, xml_path)

model_name = 'Furuta_Model.txt'
model_path = os.path.join(dirname, model_name)

# Load the model and create simulation data
model = mj.MjModel.from_xml_path(abs_path)

#print the model
mj.mj_printModel(model,model_path)

#Load the data
data = mj.MjData(model)

#Set theta1 and theta2
theta1 = (5*pi)/6
theta2 = (-3*pi)/7

data.qpos[:] = np.array([theta1, theta2])
mj.mj_forward(model, data)

#******1A Verification******
#Pull Frame c Position, Rotation Matrix, and Quaternion
body3_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "body3")

print("\n Problem 1A verification with Mujoco")
print("*************************************************")
print("Position Vector")
print(data.xpos[body3_id])

print("\n Rotation Matrix")
print(np.reshape(data.xmat[body3_id], (3,3)).round(4))

print("\n Quaternion")
print(data.xquat[body3_id])

#******1B Verification******
#Pull Frame 2 Position, Rotation Maatrix, and Quaternion
body2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "body2")

print("\n Problem 1B verification with Mujoco")
print("*************************************************")
print("Position Vector")
print(data.xpos[body2_id])

print("\n Rotation Matrix")
print(np.reshape(data.xmat[body2_id], (3,3)).round(4))

print("\n Quaternion")
print(data.xquat[body2_id])

#******1D Verification******
#Solution 1
sol1_theta1_1d = (-pi)/4   #solution 1 theta1
sol1_theta2_1d = (7*pi)/4  #solution 2 theta2

data.qpos[:] = np.array([sol1_theta1_1d, sol1_theta2_1d])
mj.mj_forward(model, data)

print("\n Problem 1D Solution 1 verification with Mujoco")
print("*************************************************")

print("Rotation Matrix")
sol1_R = np.reshape(data.xmat[body3_id], (3,3)).round(4)
print(sol1_R)

print("\n z-axis of frame 3")
sol1_z_axis = sol1_R[0:3, 2]
print(sol1_z_axis)

#Solution 2
sol2_theta1_1d = (3*pi)/4  #solution 2 theta1
sol2_theta2_1d = (pi)/4    #solution 2 theta2

data.qpos[:] = np.array([sol2_theta1_1d, sol2_theta2_1d])
mj.mj_forward(model, data)

print("\n Problem 1D Solution 2 verification with Mujoco")
print("*************************************************")

print("Rotation Matrix")
sol2_R = np.reshape(data.xmat[body3_id], (3,3)).round(4)
print(sol2_R)

print("\n z-axis of frame 3")
sol2_z_axis = sol2_R[0:3, 2]
print(sol2_z_axis)

#******1E Verification******
#Solution 1
sol1_theta1_1e = 1.07272370972679  #solution 1 theta1
sol1_theta2_1e = 1.34641832379787  #solution 2 theta2

data.qpos[:] = np.array([sol1_theta1_1e, sol1_theta2_1e])
mj.mj_forward(model, data)

print("\n Problem 1E Solution 1 verification with Mujoco")
print("*************************************************")
print("Position Vector")
print(data.xpos[body3_id]) # x-value slightly off due to constants provided by problem being rounded. Impossible to find an exact angle

#Solution 2
sol2_theta1_1e = 2.61813591673788  #solution 2 theta1
sol2_theta2_1e = 4.93676698338171  #solution 2 theta2

data.qpos[:] = np.array([sol2_theta1_1e, sol2_theta2_1e])
mj.mj_forward(model, data)

print("\n Problem 1E Solution 2 verification with Mujoco")
print("*************************************************")
print("Position Vector")
print(data.xpos[body3_id]) # x-value slightly off due to constants provided by problem being rounded. Impossible to find an exact angle

#******2A/2B Verification******
dtheta1 = 1
dtheta2 = 2

mj.mj_resetData(model, data)
mj.mj_forward(model, data)

data.qpos[:] = np.array([theta1, theta2])
data.qvel[:] = np.array([dtheta1, dtheta2])

mj.mj_forward(model, data)

twists_2AB = np.array(np.reshape(data.sensordata.copy(), (-1,6))).round(4)

print("\n Problem 2A verification with Mujoco")                 #SPATIAL LINEAR VELOCITIES DO NOT MATCH PROBLEM 2 FOR ALL VERIFICATIONS
print("*************************************************")
print("End Effector Body Twist (v,omega)")
print(twists_2AB[0])
print("\nEnd Effector Spatial Twist (v,omega)")
print(twists_2AB[1])

print("\n Problem 2B verification with Mujoco")
print("*************************************************")
print("Frame 2 Body Twist (v,omega)")
print(twists_2AB[2])
print("\nFrame 2 Spatial Twist (v,omega)")
print(twists_2AB[3])


#******2C Verification******
dtheta1_2c = -0.5
dtheta2_2c = 1.2

data.qpos[:] = np.array([theta1, theta2])
data.qvel[:] = np.array([dtheta1_2c, dtheta2_2c])
mj.mj_forward(model, data)

twists_2C = np.array(np.reshape(data.sensordata.copy(), (-1,6))).round(4)

print("\n Problem 2C verification with Mujoco")
print("*************************************************")
print("End Effector Body Twist (v,omega)")
print(twists_2C[0])

#******2D Verification******
dtheta1_2d = 0.5
dtheta2_2d = 1.2

data.qpos[:] = np.array([theta1, theta2])
data.qvel[:] = np.array([dtheta1_2d, dtheta2_2d])
mj.mj_forward(model, data)

twists_2D = np.array(np.reshape(data.sensordata.copy(), (-1,6))).round(4)

print("\n Problem 2D verification with Mujoco")
print("*************************************************")
print("End Effector Spatial Twist (v,omega)")
print(twists_2D[1])
