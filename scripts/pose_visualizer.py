import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from math import pi
from matplotlib import style
import os
from scipy.spatial.transform import Rotation as R
style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_path', dest='base_path', default='/home/asl/catkin_ws/src/oaisys/oaisys_tmp/2023-01-04-19-00-50/batch_0001/sensor_1', help='path to folder containing the poses saved by oaisys', type=str)
args = parser.parse_args()

base_path = Path(args.base_path)

# function to convert quaternions to euler angles
def quaternion_to_euler(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.array([])
y = np.array([])
z = np.array([])
u = np.array([])
v = np.array([])
w = np.array([])
for idx in range(len([entry for entry in base_path.glob('*.txt')])):
    # Load the transformation matrix from a text file
    file_name = os.path.join(base_path, f"{idx+2:04d}sensor_1_poses.txt")
    pose_vec = np.loadtxt(str(file_name))
    # Extract rotation and translation
    rotation = pose_vec[3:7]
    translation = tuple(pose_vec[0:3])

    x = np.append(x, translation[0])
    y = np.append(y, translation[1])
    z = np.append(z, translation[2])

    r = R.from_quat(rotation)
    r_matrix = r.as_matrix()
    r_rotated_matrix = np.transpose(np.array([-r_matrix[:,2], -r_matrix[:,0], r_matrix[:,1]]))
    
    rotation = R.from_euler('y', -30, degrees=True)
    r_rotated_matrix = (np.matmul(r_rotated_matrix, rotation.as_matrix()))
    
    
    r_rotated = R.from_matrix(r_rotated_matrix)
    
    r_rotated_quaterion = r_rotated.as_quat()

    

    u = np.append(u, r_rotated_matrix[0,2])
    v = np.append(v, r_rotated_matrix[1,2])
    w = np.append(w, r_rotated_matrix[2,2])
    
ax.quiver(x, y, z, u, v, w, length=5)

# Set the labels for the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot
plt.show()