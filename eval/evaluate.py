import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gt_data = pd.read_csv("../build/ground_truth.csv")
estimated_data = pd.read_csv("../build/trajectory.csv")
estimated_dr_data = pd.read_csv("../build/trajectory_dr.csv")
landmark_data = pd.read_csv("../build/landmark.csv")

fig = plt.figure(1)
ax = fig.gca(projection='3d')


# ax.scatter(landmark_data['p_x'], landmark_data['p_y'], landmark_data['p_z'])

ax.plot(estimated_dr_data['p_x'], estimated_dr_data['p_y'], estimated_dr_data['p_z'], label='dead reckoning')
ax.plot(estimated_data['p_x'], estimated_data['p_y'], estimated_data['p_z'], label='estimated')
ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], label='ground truth')

# trajectory only
# ax.set_xlim(-2.8,1.8)
# ax.set_ylim( 4.2,8.8)
# ax.set_zlim(-1.8,2.8)

boarder = 1

# ax.set_xlim(-30-boarder, 30+boarder)
# ax.set_ylim(-40-boarder, 20+boarder)
# ax.set_zlim(-20-boarder, 20+boarder)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.legend()

plt.show()

fig = plt.figure(2)

ax_q_w = plt.subplot(411)
plt.plot(gt_data['timestamp'], gt_data['q_w'], label='ground truth')
# plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['q_w'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['q_w'], label='estimated')
plt.ylabel('q_w')
plt.setp(ax_q_w.get_xticklabels(), visible=False)

plt.legend()

ax_q_x = plt.subplot(412)
plt.plot(gt_data['timestamp'], gt_data['q_x'], label='ground truth')
# plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['q_x'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['q_x'], label='estimated')
plt.ylabel('q_x')
plt.setp(ax_q_x.get_xticklabels(), visible=False)

ax_q_y = plt.subplot(413)
plt.plot(gt_data['timestamp'], gt_data['q_y'], label='ground truth')
# plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['q_y'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['q_y'], label='estimated')
plt.ylabel('q_y')
plt.setp(ax_q_y.get_xticklabels(), visible=False)

plt.subplot(414)
# plt.plot(gt_data['timestamp'], gt_data['q_z'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['q_z'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['q_z'], label='estimated')
plt.ylabel('q_z')
plt.xlabel('time [s]')
plt.show()



fig = plt.figure(3)

ax_q_w = plt.subplot(311)
plt.plot(gt_data['timestamp'], gt_data['v_x'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['v_x'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['v_x'], label='estimated')
plt.setp(ax_q_w.get_xticklabels(), visible=False)
plt.ylabel('v_x')
plt.legend()

ax_q_x = plt.subplot(312)
plt.plot(gt_data['timestamp'], gt_data['v_y'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['v_y'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['v_y'], label='estimated')
plt.ylabel('v_y')
plt.setp(ax_q_x.get_xticklabels(), visible=False)

ax_q_y = plt.subplot(313)
plt.plot(gt_data['timestamp'], gt_data['v_z'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['v_z'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['v_z'], label='estimated')
plt.ylabel('v_z')
plt.setp(ax_q_y.get_xticklabels(), visible=False)



plt.xlabel('time [s]')
plt.show()



fig = plt.figure(4)

ax_q_w = plt.subplot(311)
plt.plot(gt_data['timestamp'], gt_data['p_x'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['p_x'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['p_x'], label='estimated')
plt.setp(ax_q_w.get_xticklabels(), visible=False)
plt.ylabel('p_x')
plt.legend()

ax_q_x = plt.subplot(312)
plt.plot(gt_data['timestamp'], gt_data['p_y'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['p_y'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['p_y'], label='estimated')
plt.ylabel('p_y')
plt.setp(ax_q_x.get_xticklabels(), visible=False)

ax_q_y = plt.subplot(313)
plt.plot(gt_data['timestamp'], gt_data['p_z'], label='ground truth')
plt.plot(estimated_dr_data['timestamp'], estimated_dr_data['p_z'], label='dead reckoning')
plt.plot(estimated_data['timestamp'], estimated_data['p_z'], label='estimated')
plt.ylabel('p_z')
plt.setp(ax_q_y.get_xticklabels(), visible=False)



plt.xlabel('time [s]')
plt.show()


fig = plt.figure(5)

position_dr_error = np.zeros_like(gt_data['p_x']);
position_error = np.zeros_like(gt_data['p_x']);

for i in range(len(gt_data['p_x'])):
	position_dr_error[i] = math.sqrt( (gt_data['p_x'][i]-estimated_dr_data['p_x'][i])**2 + (gt_data['p_y'][i]-estimated_dr_data['p_y'][i])**2 + (gt_data['p_z'][i]-estimated_dr_data['p_z'][i])**2)
	position_error[i] = math.sqrt( (gt_data['p_x'][i]-estimated_data['p_x'][i])**2 + (gt_data['p_y'][i]-estimated_data['p_y'][i])**2 + (gt_data['p_z'][i]-estimated_data['p_z'][i])**2)

plt.plot(estimated_data['timestamp'], position_error, label='estimated')
plt.plot(estimated_data['timestamp'], position_dr_error, label='dead reckoning')

plt.legend()

plt.xlabel('time [s]')
plt.ylabel('error [m]')

# plt.xlim(649, 659)
# plt.ylim(0, 5)

plt.show()