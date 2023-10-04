from scipy.spatial.transform import Rotation
import numpy as np
import json
import time

class FootTrajectoryGenerator:
    def __init__(self) -> None:
        # kinematics
        self.foot_height = 0.2
        self.bottom = -0.4
        self.abduct_length = 0.037
        self.thigh_length = 0.25
        self.knee_length = 0.25
        self.num_leg = 4

        # gait
        self.init_phi_list = np.zeros(self.num_leg)
        self.phi_list = np.zeros(4)
        self.default_freq = 10.0
        self.freq_list = np.zeros(4)


    def reset(self):
        self.init_phi_list = np.array([0.0, np.pi, np.pi, 0.0]) #+ np.random.uniform(0.0, np.pi)
        self.phi_list = np.zeros(4)
        self.freq_list = np.zeros(4)

    def getJointTargets(self, t, base_mat, res_foot_pos_list, freq_list):
        theta = Rotation.from_matrix(base_mat).as_euler('xyz', degrees=False)[2]
        H_mat = Rotation.from_euler('z', theta, degrees=False).as_matrix()
        B_mat = base_mat.T@H_mat

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

            if leg_idx in [1, 3]:
                target_foot = np.array([0.0, self.abduct_length, pos]) + res_foot_pos_list[leg_idx]
                target_foot = np.clip(target_foot, [-0.1, -0.1, self.bottom - 0.1], [0.1, 0.1, self.bottom + 0.1])
                base_pos = B_mat@target_foot
                base_pos[1] = -base_pos[1]
                theta_list = self._IKSolve(base_pos)
                theta_list[0] = -theta_list[0]
            else:
                target_foot = (np.array([0.0, -self.abduct_length, pos]) + res_foot_pos_list[leg_idx])
                target_foot = np.clip(target_foot, [-0.1, -0.1, self.bottom - 0.1], [0.1, 0.1, self.bottom + 0.1])
                base_pos = B_mat@target_foot
                theta_list = self._IKSolve(base_pos)
            target_joint_list += theta_list
        return target_joint_list

    def getPhaseList(self):
        phase = np.array([np.cos(self.phi_list), np.sin(self.phi_list)]).T
        return np.ravel(phase)

    def _IKSolve(self, base_pos):
        theta1, theta2, theta3 = None, None, None
        a, b, c = base_pos
        x2 = a
        alpha = np.arctan2(c, b)
        z2 = -np.sqrt(np.clip(b**2 + c**2 - self.abduct_length**2, 0.0, np.inf))
        theta1 = alpha - np.arctan2(z2, -self.abduct_length)
        theta1 = np.mod(theta1 + np.pi, 2.0*np.pi) - np.pi
        a = (self.knee_length**2 - (x2**2 + z2**2 + self.thigh_length**2))/(2.0*self.thigh_length)
        alpha = np.arctan2(x2, z2)
        theta2 = -alpha + np.arccos(np.clip(a/np.sqrt(x2**2 + z2**2), -1.0, 1.0)) # or alpha - ~.
        theta3 = np.arctan2(x2 - self.thigh_length*np.sin(theta2), -z2 - self.thigh_length*np.cos(theta2)) - theta2
        theta3 = np.mod(theta3 + np.pi, 2.0*np.pi) - np.pi
        if theta3 < 0:
            theta2 = -alpha - np.arccos(np.clip(a/np.sqrt(x2**2 + z2**2), -1.0, 1.0))
            theta3 = np.arctan2(x2 - self.thigh_length*np.sin(theta2), -z2 - self.thigh_length*np.cos(theta2)) - theta2
            theta3 = np.mod(theta3 + np.pi, 2.0*np.pi) - np.pi
        theta2 = np.mod(theta2 + np.pi, 2.0*np.pi) - np.pi
        return [theta1, theta2, theta3]
