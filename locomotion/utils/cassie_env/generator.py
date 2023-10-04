from scipy.spatial.transform import Rotation
import numpy as np
import json
import time

class FootTrajectoryGenerator:
    def __init__(self) -> None:
        # motion
        self.default_x = 0.05
        self.foot_height = 0.3
        self.bottom = -0.75
        self.default_theta4 = -1.9

        # kinematics
        self.num_leg = 2
        self.l2 = 0.12
        self.l3 = 0.5
        self.l4 = 0.41
        self.d3 = 0.14
        self.d4 = 0.1345
        self.a = self.l2 + self.l4*np.cos(self.d4)
        self.b = self.l4*np.sin(self.d4)
        self.default_theta = np.arctan2(self.a, self.b)

        # gait
        self.init_phi_list = np.zeros(self.num_leg)
        self.phi_list = np.zeros(self.num_leg)
        self.freq_list = np.zeros(self.num_leg)
        self.default_freq = 10.0


    def reset(self):
        self.init_phi_list = np.array([0.0, np.pi]) #+ np.random.uniform(0.0, np.pi)
        self.phi_list = np.zeros(self.num_leg)
        self.freq_list = np.zeros(self.num_leg)

    def getJointTargets(self, t, freq_list=np.zeros(2)):
        self.phi_list[:] = np.mod(self.init_phi_list + (freq_list + self.default_freq)*t, 2.0*np.pi)
        self.freq_list[:] = freq_list
        k_list = 2*(self.phi_list - np.pi)/np.pi
        target_joint_list = []
        for leg_idx in range(self.num_leg):
            k = k_list[leg_idx]
            if k < 0:
                pos = self.bottom
            elif k < 1:
                pos = self.foot_height*(-2.0*(k**3) + 3.0*(k**2)) + self.bottom
            else:
                pos = self.foot_height*(2.0*(k**3) - 9.0*(k**2) + 12.0*k - 4.0) + self.bottom

            target_x = -pos
            target_y = self.default_x
            theta3 = np.arcsin((target_x**2 + target_y**2 - (self.a**2 + self.b**2 + self.l3**2))/(2*self.l3*np.sqrt(self.a**2 + self.b**2))) - self.default_theta - self.d3
            theta2 = np.arctan2(target_y, target_x) - np.arctan2(self.l3*np.sin(theta3 + self.d3) + self.l4*np.sin(self.d4), self.l2 + self.l3*np.cos(theta3 + self.d3) + self.l4*np.cos(self.d4))
            target_joint_list += [0.0, 0.0, theta2, theta3, self.default_theta4]
        return np.array(target_joint_list)

    def getPhaseList(self):
        phase = np.array([np.cos(self.phi_list), np.sin(self.phi_list)]).T
        return np.ravel(phase)
