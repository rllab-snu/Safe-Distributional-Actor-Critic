# ===== add python path ===== #
import glob
from ntpath import join
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

from utils.cassie_env.generator import FootTrajectoryGenerator

from mujoco_py import MjRenderContextOffscreen
from mujoco_py import load_model_from_path
from mujoco_py import load_model_from_xml
from mujoco_py import MjViewer
from mujoco_py import MjSim
import mujoco_py

from scipy.spatial.transform import Rotation
from collections import OrderedDict
from collections import deque
from copy import deepcopy
from gym import spaces
import numpy as np
import xmltodict
import time
import json
import gym
import sys
import cv2
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class Env(gym.Env):
    def __init__(self, use_fixed_base=False, init_base_pos=[0.0, 0.0, 1.0], init_base_quat=[1.0, 0.0, 0.0, 0.0], max_episode_length=1000) -> None:
        # =========== for simulation parameter =========== #
        self.sim_dt = 0.002
        self.contro_freq = 50.0
        self.n_substeps = int(1/(self.sim_dt*self.contro_freq))
        self.env_dt = self.sim_dt*self.n_substeps
        self.use_fixed_base = use_fixed_base
        self.gravity = np.array([0, 0, -9.8])
        self.num_legs = 2

        # for init value
        self.init_base_pos = init_base_pos
        self.init_base_quat = init_base_quat

        # for Kp & Kd of actuator
        # order: abduct, thigh, knee
        self.Kps = np.array([100,  100,  88,  96,  50]*2) 
        self.Kds = np.array([10.0, 10.0, 8.0, 9.6, 5.0]*2)

        # joint limit
        self.lower_limits = np.array([-15, -22.5, -50, -164, -140]*2)*np.pi/180.0
        self.upper_limits = np.array([22.5, 22.5, 80, -37, -30]*2)*np.pi/180.0
        self.joint_names = [
            "left_hip_roll", "left_hip_yaw", "left_hip_pitch", "left_knee", "left_foot", 
            "right_hip_roll", "right_hip_yaw", "right_hip_pitch", "right_knee", "right_foot"
        ]

        # for mujoco object
        self.model = self._loadModel(use_fixed_base=self.use_fixed_base)
        self.sim = MjSim(self.model, nsubsteps=1)
        self.viewer = None

        # get sim id
        self.robot_id = self.sim.model.body_name2id('cassie_pelvis')
        self.geom_floor_id = self.sim.model.geom_name2id('floor')
        self.geom_foot_ids = [self.sim.model.geom_name2id(f'{name}_foot') for name in ['left', 'right']]

        # joint index offset
        if self.use_fixed_base:
            self.pos_idx_offset = 0
            self.vel_idx_offset = 0
        else:
            self.pos_idx_offset = 7
            self.vel_idx_offset = 6
        # ================================================ #
        
        # foot step trajectory generator
        self.generator = FootTrajectoryGenerator()
        self.generator.foot_height = 0.0

        # environmental variables
        self.max_episode_length = max_episode_length
        self.cur_step = 0
        self.cmd_lin_vel = np.zeros(3)
        self.cmd_ang_vel = np.zeros(3)
        self.num_history = 3
        self.joint_pos_history = deque(maxlen=self.num_history)
        self.joint_vel_history = deque(maxlen=self.num_history)
        self.joint_target_history = deque(maxlen=self.num_history)

        # for gym environment
        self.state_keys = [
            'cmd_lin_vel', 'cmd_ang_vel', 'gravity_vector', 'base_lin_vel', 'base_ang_vel', 
            'joint_pos_list', 'joint_vel_list', 'phase_list', 
            'joint_pos_history', 'joint_vel_history', 'joint_target_history'
        ]
        self.num_costs = 3
        self.action_dim = len(self.lower_limits)
        self.state_dim = self.reset().shape[0]
        self.action_space = spaces.Box(
            -np.ones(self.action_dim, dtype=np.float32), 
            np.ones(self.action_dim, dtype=np.float32), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf*np.ones(self.state_dim, dtype=np.float32), 
            np.inf*np.ones(self.state_dim, dtype=np.float32), dtype=np.float32,
        )
        self.action = np.zeros(self.action_dim)
        self.action_weight = 0.75
        self.is_mirrored = False


    def reset(self):
        # reset sim & generator
        self.sim.reset()
        self.generator.reset()
        self.action = np.zeros(self.action_dim)

        # reset joint pos & vel
        init_qpos_list = [
            -6.05507670e-04, -9.46848441e-04,  6.88587941e-01,  9.43366294e-01,
            2.79908818e-03,  2.57290345e-02, -3.30741919e-01, -1.46884023e+00,
            4.99414927e-03,  1.67887314e+00, 1.56343722e-03, -1.55491322e+00, 
            1.53641059e+00, -1.65963020e+00,  6.02820781e-04,  9.20671538e-04,
            6.88739578e-01,  9.43359484e-01, -2.79838679e-03, -2.57319768e-02,
            -3.30761119e-01, -1.46842817e+00,  4.47926175e-03,  1.67906987e+00,
            1.58396908e-03, -1.55491322e+00, 1.53641059e+00, -1.65963020e+00
        ]
        if not self.use_fixed_base:
            robot_pos = np.concatenate([self.init_base_pos, self.init_base_quat], axis=0)
            self.sim.data.qpos[:self.pos_idx_offset] = robot_pos
            self.sim.data.qvel[:self.vel_idx_offset] = 0.0
        self.sim.data.qpos[self.pos_idx_offset:] = init_qpos_list
        self.sim.data.qvel[self.vel_idx_offset:] = 0.0
        self.sim.forward()

        # simulate
        joint_targets = self.generator.getJointTargets(0.0)
        joint_targets = np.clip(joint_targets, self.lower_limits, self.upper_limits)
        for _ in range(self.num_history):
            for step_idx in range(self.n_substeps):
                p_error = joint_targets - self._getJointPosList()
                d_error = self._getJointVelList()
                torque = self.Kps*p_error - self.Kds*d_error
                torque[5:7] = -torque[5:7]
                self.sim.data.ctrl[:] = torque
                self.sim.step()
            # reset variables
            self.joint_pos_history.append(p_error)
            self.joint_vel_history.append(d_error)
            self.joint_target_history.append(joint_targets)

        # reset variables
        self.pre_pos = deepcopy(self.sim.data.get_site_xpos('imu'))
        self.pre_rot = deepcopy(self.sim.data.get_site_xmat('imu'))
        self.cur_step = 0
        self.is_terminated = False
        self.cmd_lin_vel = np.array([np.random.uniform(-1.0, 1.0)] + [0.0 , 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0] + [np.random.uniform(-0.5, 0.5)])
        self.is_mirrored = (self.generator.getPhaseList()[1] < 0.0)

        # get state
        state = self._getState()
        return self._convertState(state)

    def step(self, action):
        if self.is_terminated:
            self.cur_step += 1
            state = deepcopy(self.terminal_state)
            reward = deepcopy(self.terminal_reward)
            info = deepcopy(self.terminal_info)
        else:
            state, reward, done, info = self._step(action)
            if done:
                self.is_terminated = True
                self.terminal_state = deepcopy(state)
                self.terminal_reward = deepcopy(reward)
                self.terminal_info = deepcopy(info)
        done = (self.cur_step >= self.max_episode_length)
        if done:
            info['terminal_observation'] = deepcopy(state)
        return state, reward, done, info

    def _step(self, action, **kwargs):
        # ====== before simulation step ====== #
        self.cur_step += 1
        global_t = self.cur_step*self.env_dt
        # exponential moving average
        if self.is_mirrored: 
            action = np.concatenate([action[5:], action[:5]])
        self.action = self.action*self.action_weight + np.clip(action, -1.0, 1.0)*(1.0 - self.action_weight)
        # unnormalizing
        joint_targets = self.action*(self.upper_limits - self.lower_limits)
        # get PMTG
        joint_targets += np.clip(self.generator.getJointTargets(global_t), self.lower_limits, self.upper_limits)
        joint_targets = np.clip(joint_targets, self.lower_limits, self.upper_limits)
        # rendering setting
        is_render = kwargs.get('render', False)
        if is_render: del kwargs['render']
        # ==================================== #

        # simulate
        power = 0
        for step_idx in range(self.n_substeps):
            p_error = joint_targets - self._getJointPosList()
            d_error = self._getJointVelList()
            torque = self.Kps*p_error - self.Kds*d_error
            torque[5:7] = -torque[5:7]
            self.sim.data.ctrl[:] = torque
            self.sim.step()
            power += np.sum(np.abs(self.sim.data.actuator_force*self._getJointVelList()))
            if is_render and (step_idx + 1)%5 == 0: 
                self.render(**kwargs)
        power /= self.n_substeps

        # ====== after simulation step ====== #
        self.joint_pos_history.append(p_error)
        self.joint_vel_history.append(d_error)
        self.joint_target_history.append(joint_targets)

        state = self._getState()
        self.pre_pos[:] = self.sim.data.get_site_xpos('imu')[:]
        self.pre_rot[:] = self.sim.data.get_site_xmat('imu')[:]
        reward, error = self._getReward(state, power)
        costs = self._getCosts(state)
        self.is_mirrored = (self.generator.getPhaseList()[1] < 0.0)

        body_angle = state['gravity_vector'][2]/np.linalg.norm(state['gravity_vector'])
        done = True if (self.cur_step >= self.max_episode_length) or (body_angle >= 0) else False

        info = {
            'raw_state': state,
            'costs': costs,
            'error': error,
        }
        info['num_cv'] = 1 if np.sum([float(cost >= 0.5) for cost in costs]) >= 1.0 else 0
        # =================================== #
        return self._convertState(state), reward, done, info

    def render(self, mode='human', size=(512, 512), **kwargs):
        if mode == 'rgb_array':
            # img = self.sim.render(*size, camera_name="frontview")
            # img = img[::-1,:,:]
            # img2 = self.sim.render(*size, camera_name="fixed")
            # img2 = img2[::-1,:,:]
            # img = np.concatenate([img, img2], axis=1)
            # return img
            img = self.sim.render(*size, camera_name="track")
            img = img[::-1,:,:]
            return img
        else:
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
                self._viewerSetup(self.viewer)
            self.viewer.render()
            self._addMarker()

    def close(self):
        return

    def _loadModel(self, use_fixed_base=False):
        # load xml file
        robot_base_path = f"{ABS_PATH}/models/mjcf.xml"
        with open(robot_base_path) as f:
            robot_base_xml = f.read()
        xml = xmltodict.parse(robot_base_xml)
        body_xml = xml['mujoco']['worldbody']['body']
        if type(body_xml) != OrderedDict:
            body_xml = body_xml[0]

        # for time interval
        if type(xml['mujoco']['option']) == list:
            for option in xml['mujoco']['option']:
                if '@timestep' in option.keys():
                    option['@timestep'] = self.sim_dt
                    break
        else:
            option = xml['mujoco']['option']
            option['@timestep'] = self.sim_dt

        # for base fix
        if use_fixed_base:
            del body_xml['joint']
        body_xml['@quat'] = ' '.join([f"{i}" for i in self.init_base_quat])
        body_xml['@pos'] = ' '.join([f"{i}" for i in self.init_base_pos])

        # convert xml to string & load model
        xml['mujoco']['compiler']['@meshdir'] = f'{ABS_PATH}/models/meshes'
        xml_string = xmltodict.unparse(xml)
        model = load_model_from_xml(xml_string)
        return model

    def _getJointPosList(self):
        joint_pos_list = np.array([self.sim.data.get_joint_qpos(joint_name) for joint_name in self.joint_names])
        joint_pos_list[5:7] = -joint_pos_list[5:7]
        return joint_pos_list

    def _getJointVelList(self):
        joint_vel_list = np.array([self.sim.data.get_joint_qvel(joint_name) for joint_name in self.joint_names])            
        joint_vel_list[5:7] = -joint_vel_list[5:7]
        return joint_vel_list

    def _viewerSetup(self, viewer):
        viewer.cam.trackbodyid = self.robot_id
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

    def _getState(self):
        state = {}
        state['cmd_lin_vel'] = self.cmd_lin_vel
        state['cmd_ang_vel'] = self.cmd_ang_vel

        base_pos = self.sim.data.get_site_xpos('imu')
        base_mat = self.sim.data.get_site_xmat('imu')
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        rot_mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()

        state['base_height'] = base_pos[2:]

        gravity_vector = base_mat@self.gravity
        state['gravity_vector'] = rot_mat.T@gravity_vector

        ang_diff = base_mat@self.pre_rot.T
        base_lin_vel = (base_pos - self.pre_pos)/self.env_dt
        base_ang_vel = np.array([ang_diff[2, 1], ang_diff[0, 2], ang_diff[1, 0]])/self.env_dt
        state['base_lin_vel'] = rot_mat.T@base_lin_vel
        state['base_ang_vel'] = rot_mat.T@base_ang_vel

        state['joint_pos_list'] = self._getJointPosList()
        state['joint_vel_list'] = self._getJointVelList()

        state['phase_list'] = self.generator.getPhaseList()
        state['base_freq'] = np.array([self.generator.default_freq])
        state['freq_list'] = deepcopy(self.generator.freq_list)

        state['joint_pos_history'] = np.concatenate(list(self.joint_pos_history)[-2:])
        state['joint_vel_history'] = np.concatenate(list(self.joint_vel_history)[-2:])
        state['joint_target_history'] = np.concatenate(list(self.joint_target_history)[-2:])

        contact_list = np.zeros(self.num_legs)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            for geom_toe_idx, geom_toe_id in enumerate(self.geom_foot_ids):
                if contact.geom1 == geom_toe_id or contact.geom2 == geom_toe_id:
                    if contact.geom1 != contact.geom2:
                        contact_list[geom_toe_idx] = 1.0
        state['contact_list'] = contact_list

        collision = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.geom1 == contact.geom2:
                continue
            condition1 = (contact.geom1 == self.geom_floor_id) and (not contact.geom2 in self.geom_foot_ids)
            condition2 = (contact.geom2 == self.geom_floor_id) and (not contact.geom1 in self.geom_foot_ids)
            if condition1 or condition2:
                collision = True
                break
        state['collision'] = collision
        return state

    def _convertState(self, state):
        flatten_state = []
        for key in self.state_keys:
            flatten_state.append(state[key])
        '''
        state_keys:
            'cmd_lin_vel', 'cmd_ang_vel', 'gravity_vector', 'base_lin_vel', 'base_ang_vel', 
            'joint_pos_list', 'joint_vel_list', 'phase_list', 
            'joint_pos_history', 'joint_vel_history', 'joint_target_history'

        cmd_lin_vel: [0, -1, 2]
        cmd_ang_vel: [-0, 1, -2] + 3
        gravity_vector: [0, -1, 2] + 6
        base_lin_vel: [0, -1, 2] + 9
        base_ang_vel: [-0, 1, -2] + 12
        joint_pos_list: [5:10, 0:5] + 15
        joint_vel_list: [5:10, 0:5] + 25
        phase_list: [2:4, 0:2] + 35
        joint_pos_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 39
        joint_vel_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 69
        joint_target_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 99
        '''
        state = np.concatenate(flatten_state)
        if self.is_mirrored:
            idx = 0 # cmd_lin_vel
            state[idx + 1] = -state[idx + 1]
            idx += 3 # cmd_ang_vel
            state[idx + 0] = -state[idx + 0]
            state[idx + 2] = -state[idx + 2]
            idx += 3 # gravity vector
            state[idx + 1] = -state[idx + 1]
            idx += 3 # base_lin_vel
            state[idx + 1] = -state[idx + 1]
            idx += 3 # base_ang_vel
            state[idx + 0] = -state[idx + 0]
            state[idx + 2] = -state[idx + 2]
            idx += 3 # joint_pos_list
            temp = deepcopy(state[(idx+5):(idx+10)])
            state[(idx+5):(idx+10)] = state[(idx+0):(idx+5)]
            state[(idx+0):(idx+5)] = temp[:]
            idx += 10 # joint_vel_list
            temp = deepcopy(state[(idx+5):(idx+10)])
            state[(idx+5):(idx+10)] = state[(idx+0):(idx+5)]
            state[(idx+0):(idx+5)] = temp[:]
            idx += 10 # phase_list
            temp = deepcopy(state[(idx+2):(idx+4)])
            state[(idx+2):(idx+4)] = state[(idx+0):(idx+2)]
            state[(idx+0):(idx+2)] = temp[:]
            idx += 4 # joint_pos_history
            for ii in range(self.num_history - 1):
                temp = deepcopy(state[(idx+5+ii*10):(idx+10+ii*10)])
                state[(idx+5+ii*10):(idx+10+ii*10)] = state[(idx+ii*10):(idx+5+ii*10)]
                state[(idx+ii*10):(idx+5+ii*10)] = temp[:]
            idx += 10*(self.num_history - 1) # joint_vel_history
            for ii in range(self.num_history - 1):
                temp = deepcopy(state[(idx+5+ii*10):(idx+10+ii*10)])
                state[(idx+5+ii*10):(idx+10+ii*10)] = state[(idx+ii*10):(idx+5+ii*10)]
                state[(idx+ii*10):(idx+5+ii*10)] = temp[:]
            idx += 10*(self.num_history - 1) # joint_target_history
            for ii in range(self.num_history - 1):
                temp = deepcopy(state[(idx+5+ii*10):(idx+10+ii*10)])
                state[(idx+5+ii*10):(idx+10+ii*10)] = state[(idx+ii*10):(idx+5+ii*10)]
                state[(idx+ii*10):(idx+5+ii*10)] = temp[:]
        return state

    def _getReward(self, state, power):
        ang_vel_error = (self.cmd_ang_vel[2] - state['base_ang_vel'][2])**2
        lin_vel_error = np.sum(np.square(state['base_lin_vel'][:2] - self.cmd_lin_vel[:2]))
        error = ang_vel_error + lin_vel_error
        power_reward = -1e-3*power # second best
        reward = 0.1*(-error + power_reward)
        return reward, error

    def _getCosts(self, state):
        costs = []

        # for body angle constraint
        a = -np.cos(15.0*(np.pi/180.0))
        x = state['gravity_vector'][2]/np.linalg.norm(state['gravity_vector'])
        costs.append(1.0 if x > a else 0.0)

        # for height
        a = 0.7
        x = state['base_height'][0]
        costs.append(1.0 if x < a else 0.0)

        # swing timing
        cost = 0.0
        for leg_idx in range(self.num_legs):
            cos_phase, sin_phase = state['phase_list'][2*leg_idx:2*(leg_idx+1)]
            if sin_phase < 0.0: # swing phase
                cost += 1.0 if state['contact_list'][leg_idx] else 0.0
            else: # stance phase
                cost += 0.0 if state['contact_list'][leg_idx] else 1.0
        cost /= self.num_legs
        costs.append(cost)

        # assert len(costs) == self.num_costs
        return costs

    def _addMarker(self):
        base_pos = self.sim.data.get_site_xpos('imu')
        base_mat = self.sim.data.get_site_xmat('imu')
        pos = base_pos + np.array([0.0, 0.0, 0.4])
        size = np.clip(self.cmd_ang_vel[2], -0.5, 0.5)/0.5
        self.viewer.add_marker(
            pos=pos,
            size=np.array([0.01, 0.01, size*0.5]),
            mat=np.eye(3),
            type=mujoco_py.const.GEOM_ARROW,
            rgba=np.array([1, 1, 0, 1]),
            label=''
        )
        size = np.clip(self.cmd_lin_vel[0], -0.5, 0.5)/0.5
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()@Rotation.from_rotvec([0.0, np.pi/2, 0.0]).as_matrix()
        self.viewer.add_marker(
            pos=pos,
            size=np.array([0.01, 0.01, size*0.5]),
            mat=mat,
            type=mujoco_py.const.GEOM_ARROW,
            rgba=np.array([0, 0, 1, 1]),
            label=''
        )
        sin_phase = self.generator.getPhaseList()[1]
        if sin_phase < 0.0:
            color = np.array([1, 0, 0, 1])
        else:
            color = np.array([0, 1, 0, 1])
        new_pos = pos + np.array([0.0, 0.0, 0.5])
        self.viewer.add_marker(
            pos=new_pos,
            size=np.ones(3)*0.1,
            mat=np.eye(3),
            type=mujoco_py.const.GEOM_SPHERE,
            rgba=color,
            label=''
        )


if __name__ == "__main__":
    # env = Env(use_fixed_base=True, init_base_pos=[0, 0, 1.0])
    env = Env(use_fixed_base=False)

    for i in range(10):
        env.reset()
        start_t = time.time()
        global_t = 0.0
        elapsed_t = 0.0
        action = np.zeros(env.action_space.shape[0])
        # action[0] = action[5+0] = 0.5
        for i in range(100):
            # action = env.action_space.sample()
            # s, r, d, info = env.step(action*0.5)
            s, r, d, info = env.step(action)
            env.render()
            global_t += env.env_dt

            elapsed_t = time.time() - start_t
            if elapsed_t < global_t:
                time.sleep(global_t - elapsed_t)

            if d: break
        # print(env.sim.data.qpos)
        # exit()

    #         sys.stdout.write("\rsim time : {:.3f} s, real time : {:.3f} s".format(global_t, elapsed_t))
    #         sys.stdout.flush()
    # sys.stdout.write(f"\rgoodbye.{' '*50}\n")
