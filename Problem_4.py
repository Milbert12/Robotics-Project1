import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
import mujoco.viewer
import os
from math import pi
import time

def sgn(x):
    return np.piecewise(x, [x < 0, x >= 0], [-1, 1])

xml_path = "Furuta_Model.xml"
dirname = os.path.dirname(__file__)
abs_path = os.path.join(dirname, xml_path)

model_name = 'Furuta_Model.txt'
model_path = os.path.join(dirname, model_name)

# Load the model and create simulation data
model = mj.MjModel.from_xml_path(abs_path)

#Load the data
data = mj.MjData(model)


sim_time = 15.0 #15 second simulation time
start_time = time.time()
actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "joint1motor")
body3_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "body3")
body1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "body1")

height = np.array([])
x_axis = np.array([])
b3_body_lin = np.array([])
b3_spatial_ang = np.array([])
b1_spatial_ang = np.array([])

while ((time.time() - start_time) < sim_time):
    step_start = time.time()
    
    torque = -15*sgn(data.qvel[1].copy())
    data.ctrl[actuator_id] = torque

    twists = np.array(np.reshape(data.sensordata.copy(), (-1, 6))).round(4)

    mj.mj_step(model, data)

    x_axis = np.append(x_axis, time.time() - start_time)
    height = np.append(height, data.xpos[body3_id][2])
    b3_body_lin = np.append(b3_body_lin, twists[0][0:3])
    b3_spatial_ang = np.append(b3_spatial_ang, twists[1][3:6])
    b1_spatial_ang = np.append(b1_spatial_ang, twists[4][3:6])
    
    time_until_next_step = model.opt.timestep - (time.time() - step_start) #this limits the while loop to only run every 0.002seconds
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

#**************4A****************
fig, h_4a = plt.subplots()
h_4a.plot(x_axis, height)

h_4a.set(xlabel='Time (s)', ylabel='Height',
       title='End Effector Height with Respect to World Frame')
h_4a.grid()

fig.savefig("height.png")


#***************4B****************
b3_body_lin = np.reshape(b3_body_lin, (-1,3))

x_b3_body_lin = b3_body_lin[:,0]
y_b3_body_lin = b3_body_lin[:,1]
z_b3_body_lin = b3_body_lin[:,2]

N_b3_body_lin = np.linalg.norm(b3_body_lin, axis=1)

fig, (linx_4b, liny_4b, linz_4b) = plt.subplots(3)

linx_4b.plot(x_axis, x_b3_body_lin)
linx_4b.set(ylabel= 'x', title='End Effector Body Linear Velocity x y z')

liny_4b.plot(x_axis, y_b3_body_lin)
liny_4b.set(ylabel= 'y')

linz_4b.plot(x_axis, z_b3_body_lin)
linz_4b.set(ylabel= 'z', xlabel='Time (s)')


fig.savefig("Lin-Velocity.png")


fig, (linN_4b) = plt.subplots()
linN_4b.plot(x_axis, N_b3_body_lin)

linN_4b.set(xlabel='Time (s)', ylabel='Linear Velocity Norm',
       title='End Effector Body Linear Velocity')
linN_4b.grid()

fig.savefig("Lin-Velocity Norm.png")

#***********4C************
b3_spatial_ang = np.reshape(b3_spatial_ang, (-1,3))
b1_spatial_ang = np.reshape(b1_spatial_ang, (-1,3))

x_b3_spatial_ang = b3_spatial_ang[:,0]
y_b3_spatial_ang = b3_spatial_ang[:,1]
z_b3_spatial_ang = b3_spatial_ang[:,2]
N_b3_spatial_ang = np.linalg.norm(b3_spatial_ang, axis=1)


x_b1_spatial_ang = b1_spatial_ang[:,0]
y_b1_spatial_ang = b1_spatial_ang[:,1]
z_b1_spatial_ang = b1_spatial_ang[:,2]
N_b1_spatial_ang = np.linalg.norm(b1_spatial_ang, axis=1)

fig, (linx_4c, liny_4c, linz_4c) = plt.subplots(3)

linx_4c.plot(x_b1_spatial_ang, x_b3_spatial_ang)
linx_4c.set(ylabel= 'Frame 3 x', title='End Effector vs Frame 1 Spatial Angular Velocity x y z')

liny_4c.plot(y_b1_spatial_ang, y_b3_spatial_ang)
liny_4c.set(ylabel= 'Frame 3 y')

linz_4c.plot(z_b1_spatial_ang, z_b3_spatial_ang)
linz_4c.set(ylabel= 'Frame 3 z', xlabel='Frame 1 x, y, z')


fig.savefig("Spatial Angular Velocity xyz.png")

fig, (linN_4c) = plt.subplots()
linN_4c.plot(N_b1_spatial_ang, N_b3_spatial_ang)

linN_4c.set(xlabel='Frame 1 Norm', ylabel='Frame 2 Norm',
       title='End Effector vs Frame 1 Spatial Angular Velocity')
linN_4c.grid()

fig.savefig("Spatial Angular Velocity Norm.png")

plt.show()
