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

from utils.laikago_env.generator import FootTrajectoryGenerator

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
    def __init__(self, use_fixed_base=False, init_base_pos=[0.0, 0.0, 0.5], init_base_quat=[1.0, 0.0, 0.0, 0.0], max_episode_length=1000) -> None:
        # =========== for simulation parameter =========== #
        self.sim_dt = 0.002
        self.contro_freq = 50.0
        self.n_substeps = int(1/(self.sim_dt*self.contro_freq))
        self.env_dt = self.sim_dt*self.n_substeps
        self.use_fixed_base = use_fixed_base
        self.gravity = np.array([0, 0, -9.8])
        self.num_legs = 4

        # for init value
        self.init_base_pos = init_base_pos
        self.init_base_quat = init_base_quat

        # for Kp & Kd of actuator
        # order: abduct, thigh, knee
        self.Kp_list = [100.0, 100.0, 100.0]
        self.Kd_list = [5.0, 5.0, 5.0]

        # joint limit
        self.lower_limits = np.array([-1.0, -3.0, -2.0]*self.num_legs)
        self.upper_limits = np.array([1.0, 1.5, 2.5]*self.num_legs)

        # for mujoco object
        self.model = self._loadModel(use_fixed_base=self.use_fixed_base)
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = None

        # get sim id
        self.robot_id = self.sim.model.body_name2id('torso')
        self.geom_floor_id = self.sim.model.geom_name2id('floor')
        self.geom_toe_ids = [self.sim.model.geom_name2id(f"{name}_toe") for name in ['fr', 'fl', 'hr', 'hl']]
        self.geom_knee_ids = [self.sim.model.geom_name2id(f"{name}_knee") for name in ['fr', 'fl', 'hr', 'hl']]
        self.geom_total_ids = self.geom_toe_ids + self.geom_knee_ids

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
        joint_targets = self.generator.getJointTargets(0.0, np.eye(3), np.zeros((4, 3)), np.zeros(4))
        if not self.use_fixed_base:
            robot_pos = np.concatenate([self.init_base_pos, self.init_base_quat], axis=0)
            self.sim.data.set_joint_qpos('world_to_torso_j', robot_pos)
            self.sim.data.set_joint_qvel('world_to_torso_j', np.zeros(6))
        self.sim.data.qpos[self.pos_idx_offset:] = joint_targets
        self.sim.data.qvel[self.vel_idx_offset:] = np.zeros(12)
        # if qpos or qvel have been modified directly, the user is required to call forward()
        self.sim.forward()

        # simulate for stable init pose
        for _ in range(100):
            self.sim.data.ctrl[:] = joint_targets
            self.sim.step()

        # reset variables
        self.cur_step = 0
        self.is_terminated = False
        self.cmd_lin_vel = np.array([np.random.uniform(-1.0, 2.0)] + [0.0 , 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0] + [np.random.uniform(-0.5, 0.5)])
        for _ in range(self.num_history):
            self.joint_pos_history.append(np.zeros(self.action_dim))
            self.joint_vel_history.append(np.zeros(self.action_dim))
            self.joint_target_history.append(joint_targets)
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

    def _step(self, action):
        # ====== before simulation step ====== #
        self.cur_step += 1
        if self.is_mirrored: 
            action = np.concatenate([action[3:6], action[0:3], action[9:12], action[6:9]])
        # exponential moving average
        self.action = self.action*self.action_weight + np.clip(action, -1.0, 1.0)*(1.0 - self.action_weight)
        global_t = self.cur_step*self.env_dt
        target_joint_list = np.clip(self.generator.getJointTargets(
            global_t, np.eye(3), np.zeros((4, 3)), np.zeros(4)
        ), self.lower_limits, self.upper_limits)
        target_joint_list += self.action*(self.upper_limits - self.lower_limits)
        target_joint_list = np.clip(target_joint_list, self.lower_limits, self.upper_limits)
        self.sim.data.ctrl[:] = target_joint_list
        self.sim.data.ctrl[3] *= -1.0
        self.sim.data.ctrl[9] *= -1.0
        # ==================================== #

        # simulate
        self.sim.step()

        # ====== after simulation step ====== #
        self.joint_pos_history.append(target_joint_list - self._getJointPosList())
        self.joint_vel_history.append(self._getJointVelList())
        self.joint_target_history.append(target_joint_list)

        state = self._getState()
        reward, error = self._getReward(state)
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

        # ========== for actuator ========== #
        # for Kd setting
        for abduct_idx in range(4):
            abduct_xml = body_xml['body'][abduct_idx]
            abduct_joint = abduct_xml['joint']
            thigh_joint = abduct_xml['body']['joint']
            knee_joint = abduct_xml['body']['body']['joint']            
            joint_list = [abduct_joint, thigh_joint, knee_joint]
            for joint_idx in range(len(joint_list)):
                joint = joint_list[joint_idx]
                joint['@damping'] = self.Kd_list[joint_idx]
        # for Kp setting
        for actuator in xml['mujoco']['actuator']['position']:
            act_name = actuator['@name']
            if 'torso_to_abduct' in act_name:
                actuator['@kp'] = self.Kp_list[0]
            elif 'abduct_to_thigh' in act_name:
                actuator['@kp'] = self.Kp_list[1]
            elif 'thigh_to_knee' in act_name:
                actuator['@kp'] = self.Kp_list[2]
            else:
                raise NameError('The xml file has wrong actuator name.')
        # ================================== #

        # convert xml to string & load model
        xml['mujoco']['compiler']['@meshdir'] = f'{ABS_PATH}/models/meshes'
        xml_string = xmltodict.unparse(xml)
        model = load_model_from_xml(xml_string)
        return model

    def _getJointPosList(self):
        joint_pos_list = np.array(self.sim.data.qpos[self.pos_idx_offset:])
        joint_pos_list[3] *= -1.0
        joint_pos_list[9] *= -1.0
        return joint_pos_list

    def _getJointVelList(self):
        joint_vel_list = np.array(self.sim.data.qvel[self.vel_idx_offset:])        
        joint_vel_list[3] *= -1.0
        joint_vel_list[9] *= -1.0
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

        base_pos = self.sim.data.get_site_xpos('robot')
        base_mat = self.sim.data.get_site_xmat('robot')
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        rot_mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()

        state['base_height'] = base_pos[2:]

        gravity_vector = base_mat@self.gravity
        state['gravity_vector'] = rot_mat.T@gravity_vector

        base_lin_vel = rot_mat.T@self.sim.data.get_site_xvelp('robot')
        base_ang_vel = rot_mat.T@self.sim.data.get_site_xvelr('robot')
        state['base_lin_vel'] = base_lin_vel
        state['base_ang_vel'] = base_ang_vel

        state['joint_pos_list'] = self._getJointPosList()
        state['joint_vel_list'] = self._getJointVelList()

        state['phase_list'] = self.generator.getPhaseList()
        state['base_freq'] = np.array([self.generator.default_freq])
        state['freq_list'] = deepcopy(self.generator.freq_list)

        state['joint_pos_history'] = np.concatenate(list(self.joint_pos_history)[-2:])
        state['joint_vel_history'] = np.concatenate(list(self.joint_vel_history)[-2:])
        state['joint_target_history'] = np.concatenate(list(self.joint_target_history)[-2:])

        contact_list = np.zeros(4)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            for geom_toe_idx, geom_toe_id in enumerate(self.geom_toe_ids):
                if contact.geom1 == geom_toe_id or contact.geom2 == geom_toe_id:
                    if contact.geom1 != contact.geom2:
                        contact_list[geom_toe_idx] = 1.0
        state['contact_list'] = contact_list

        # collision = False
        # for i in range(self.sim.data.ncon):
        #     contact = self.sim.data.contact[i]
        #     if contact.geom1 == contact.geom2:
        #         continue
        #     condition1 = (contact.geom1 == self.geom_floor_id) and (not contact.geom2 in self.geom_total_ids)
        #     condition2 = (contact.geom2 == self.geom_floor_id) and (not contact.geom1 in self.geom_total_ids)
        #     if condition1 or condition2:
        #         collision = True
        #         break
        # state['collision'] = collision
        return state

    def _convertState(self, state):
        flatten_state = []
        for key in self.state_keys:
            flatten_state.append(state[key])
        state = np.concatenate(flatten_state)
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
        joint_pos_list: [3:6, 0:3, 9:12, 6:9] + 15
        joint_vel_list: [3:6, 0:3, 9:12, 6:9] + 27
        phase_list: [2:4, 0:2] + 39
        joint_pos_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 43
        joint_vel_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 79
        joint_target_history: [5:10, 0:5, 15:20, 10:15, 25:30, 20:25] + 115
        151
        '''
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
            new_state = np.concatenate([
                state[(idx+3):(idx+6)],
                state[(idx+0):(idx+3)],
                state[(idx+9):(idx+12)],
                state[(idx+6):(idx+9)],
            ])
            state[idx:(idx+12)] = new_state[:]
            idx += 12 # joint_vel_list
            new_state = np.concatenate([
                state[(idx+3):(idx+6)],
                state[(idx+0):(idx+3)],
                state[(idx+9):(idx+12)],
                state[(idx+6):(idx+9)],
            ])
            state[idx:(idx+12)] = new_state[:]
            idx += 12 # phase_list
            temp = deepcopy(state[(idx+2):(idx+4)])
            state[(idx+2):(idx+4)] = state[(idx+0):(idx+2)]
            state[(idx+0):(idx+2)] = temp[:]
            idx += 4 # joint_pos_history
            for ii in range(self.num_history - 1):
                new_state = np.concatenate([
                    state[(idx+3):(idx+6)],
                    state[(idx+0):(idx+3)],
                    state[(idx+9):(idx+12)],
                    state[(idx+6):(idx+9)],
                ])
                state[idx:(idx+12)] = new_state[:]
                idx += 12
            # joint_vel_history
            for ii in range(self.num_history - 1):
                new_state = np.concatenate([
                    state[(idx+3):(idx+6)],
                    state[(idx+0):(idx+3)],
                    state[(idx+9):(idx+12)],
                    state[(idx+6):(idx+9)],
                ])
                state[idx:(idx+12)] = new_state[:]
                idx += 12
            # joint_target_history
            for ii in range(self.num_history - 1):
                new_state = np.concatenate([
                    state[(idx+3):(idx+6)],
                    state[(idx+0):(idx+3)],
                    state[(idx+9):(idx+12)],
                    state[(idx+6):(idx+9)],
                ])
                state[idx:(idx+12)] = new_state[:]
                idx += 12
        return state

    def _getReward(self, state):
        ang_vel_error = (self.cmd_ang_vel[2] - state['base_ang_vel'][2])**2
        lin_vel_error = np.sum(np.square(state['base_lin_vel'][:2] - self.cmd_lin_vel[:2]))
        error = ang_vel_error + lin_vel_error
        power_reward = -1e-3*np.sum(np.abs(self.sim.data.actuator_force*state['joint_vel_list']))
        reward = 0.1*(-error + power_reward)
        return reward, error

    def _getCosts(self, state):
        costs = []

        # for body angle constraint
        a = -np.cos(15.0*(np.pi/180.0))
        x = state['gravity_vector'][2]/np.linalg.norm(state['gravity_vector'])
        costs.append(1.0 if x > a else 0.0)

        # for height
        a = 0.35
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
        base_pos = self.sim.data.get_site_xpos('robot')
        base_mat = self.sim.data.get_site_xmat('robot')
        pos = base_pos + np.array([0.0, 0.0, 0.4])
        size = np.clip(self.cmd_ang_vel[2], -0.5, 0.5)/0.5
        self.viewer.add_marker(
            pos=pos,
            size=np.array([0.01, 0.01, size*0.5]),
            mat=np.eye(3),
            type=mujoco_py.const.GEOM_ARROW,
            rgba=np.array([1, 0, 0, 1]),
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


if __name__ == "__main__":
    env = Env(use_fixed_base=True)
    # env = Env(use_fixed_base=False)

    for i in range(10):
        env.reset()
        start_t = time.time()
        global_t = 0.0
        elapsed_t = 0.0
        # action = np.zeros(env.action_space.shape[0])
        action = np.array([-1.0, 0.0, 0.0]*4)
        for i in range(1000):
            s, r, d, info = env.step(action)
            env.render()
            global_t += env.env_dt

            elapsed_t = time.time() - start_t
            if elapsed_t < global_t:
                time.sleep(global_t - elapsed_t)

            if d: break
            
            sys.stdout.write("\rsim time : {:.3f} s, real time : {:.3f} s".format(global_t, elapsed_t))
            sys.stdout.flush()
    sys.stdout.write(f"\rgoodbye.{' '*50}\n")
