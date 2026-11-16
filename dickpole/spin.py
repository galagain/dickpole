"""Helicopter task: passive 3D rotor driven only through muscle-driven pelvis motion."""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium import spaces

from .environment import JOINTS, MuscleHumanoid, model_xml

SIDE_JOINTS = ('lumbar_roll', 'hip_roll_l', 'ankle_roll_l', 'hip_roll_r', 'ankle_roll_r')


def spin_model_xml():
    root = ET.fromstring(model_xml())
    root.set('model', 'DickPole helicopter')

    def joint(body_name, **attributes):
        body = root.find(f".//body[@name='{body_name}']")
        body.insert(len(body.findall('joint')), ET.Element('joint', attributes))

    joint('pelvis', name='root_y', type='slide', axis='0 1 0', limited='false', damping='0')
    joint('pelvis', name='root_roll', type='hinge', axis='1 0 0', limited='false', damping='0')
    joint('torso', name='lumbar_roll', axis='1 0 0', range='-.55 .55', stiffness='10')
    for side in ('l', 'r'):
        joint(f'thigh_{side}', name=f'hip_roll_{side}', axis='1 0 0', range='-.6 .6')
        joint(f'foot_{side}', name=f'ankle_roll_{side}', axis='1 0 0', range='-.6 .6')
    rotor = root.find(".//body[@name='pole_0']")
    # An anterior stand-off keeps full frontal revolutions clear of the torso.
    rotor.set('pos', '.28 0 -.02')
    rotor.find("joint[@name='pole_0_joint']").set('axis', '1 0 0')
    joint('pole_0', name='pole_splay', axis='0 1 0', range='-.2 .2', stiffness='1.2', damping='.02')
    pelvis = root.find(".//body[@name='pelvis']")
    ET.SubElement(pelvis, 'geom', name='rotor_attachment', type='capsule',
                  fromto='.10 0 -.02 .28 0 -.02', size='.012', mass='.01',
                  contype='0', conaffinity='0', rgba='.6 .65 .7 1')
    tendons, actuators = root.find('tendon'), root.find('actuator')
    for name in SIDE_JOINTS:
        for direction, sign in [('flexor', -1), ('extensor', 1)]:
            identifier = f'{name}_{direction}'
            tendon = ET.SubElement(tendons, 'fixed', name=identifier)
            ET.SubElement(tendon, 'joint', joint=name, coef=str(sign*.04))
            ET.SubElement(actuators, 'muscle', name=identifier, tendon=identifier,
                          force='6000', lengthrange='-.12 .12', range='.75 1.05', timeconst='.01 .04')
    return ET.tostring(root, encoding='unicode')


class SpinHumanoid(MuscleHumanoid):
    """Two commands: lateral pelvis velocity and vertical pelvis velocity.

    The skeleton moves in 3D; a passive front-facing revolute joint and passive
    out-of-plane bending produce the helicopter motion. No rotor motor exists.
    """
    def __init__(self, duration=20., direction=1, target_speed=2*math.pi):
        if direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1')
        self.model = mujoco.MjModel.from_xml_string(spin_model_xml())
        self.data = mujoco.MjData(self.model)
        self.duration, self.direction, self.target_speed = duration, direction, target_speed
        self.tilt_rate = 2.4
        for name in ('_tip_jac', '_base_jac', '_com_jac', '_rotation_jac'):
            setattr(self, name, np.zeros((3, self.model.nv)))
        self.qids = {self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)}
        self.dofs = {self.model.joint(i).name: int(self.model.jnt_dofadr[i]) for i in range(self.model.njnt)}
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, self.observe().shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.target_y = self.target_roll = 0.
        self.spin_action = np.zeros(2)
        self.signed_angle = self.streak_angle = self.best_streak_angle = 0.
        self.previous_rotor_angle = self.stall_time = 0.
        self.contact_time = 0.
        return super().reset(seed=seed, options=options)

    def observe(self):
        angle, speed = self.pole_state()
        return np.r_[self.data.qpos, self.data.qvel*.2, self.data.act,
                     self.target_y, self.target_height, self.spin_action,
                     math.sin(angle), math.cos(angle), speed*.1,
                     self.direction, self.target_speed*.1, self.data.time/20.].astype(np.float32)

    def reflex(self):
        super().reflex()
        q, v, m, d = self.data.qpos, self.data.qvel, self.model, self.data
        roll = q[self.qids['root_roll']]
        roll_speed = v[self.dofs['root_roll']]
        foot_center = (d.site('sole_l').xpos[1]+d.site('sole_r').xpos[1])/2
        error_y = q[self.qids['root_y']]-foot_center-self.target_y
        lean = math.asin(np.clip(-self.target_y/max(.4, self.target_height-.1), -.3, .3))
        balance = roll-self.target_roll+.2*roll_speed-error_y-.8*v[self.dofs['root_y']]
        hip, ankle = lean-self.target_roll, -lean+balance
        lumbar = np.clip(-.3*roll-.7*self.target_roll+3*self.target_y, -.5, .5)
        targets = [lumbar, hip, ankle, hip, ankle]
        for j, (name, target) in enumerate(zip(SIDE_JOINTS, targets), len(JOINTS)):
            torque = 350*(target-q[self.qids[name]])-30*v[self.dofs[name]]
            for k, sign in ((2*j, -1), (2*j+1, 1)):
                gain = mujoco.mju_muscleGain(d.actuator_length[k], d.actuator_velocity[k],
                                             m.actuator_lengthrange[k], m.actuator_acc0[k], m.actuator_gainprm[k, :9])
                d.ctrl[k] = min(1., .015+max(0., -sign*torque)/max(1., abs(gain*.04)))

    def pole_state(self):
        m, d = self.model, self.data
        base = d.body('pole_0').xpos
        delta = d.site('pole_tip').xpos-base
        mujoco.mj_jacSite(m, d, self._tip_jac, None, m.site('pole_tip').id)
        mujoco.mj_jacBody(m, d, self._base_jac, None, m.body('pole_0').id)
        mujoco.mj_jacBody(m, d, None, self._rotation_jac, m.body('pelvis').id)
        omega = self._rotation_jac @ d.qvel
        velocity = (self._tip_jac-self._base_jac) @ d.qvel-np.cross(omega, delta)
        rotation = d.body('pelvis').xmat.reshape(3, 3).T
        local, local_velocity = rotation @ delta, rotation @ velocity
        denominator = max(1e-10, local[1]**2+local[2]**2)
        angle = math.atan2(local[1], -local[2])
        speed = (-local[2]*local_velocity[1]+local[1]*local_velocity[2])/denominator
        return angle, float(speed)

    def _rotation_progress(self, change, speed, free):
        progress = self.direction*change
        self.signed_angle += progress
        self.stall_time = self.stall_time+self.dt if self.direction*speed < .2 else 0.
        if not free or progress < -.01 or self.stall_time > .4:
            self.streak_angle = 0.
        else:
            self.streak_angle = max(0., self.streak_angle+progress)
        self.best_streak_angle = max(self.best_streak_angle, self.streak_angle)
        # Reverse motion always incurs its cost, including during contact.
        return min(progress, 0.)+(max(progress, 0.) if free else 0.)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Expected finite [lateral_velocity, vertical_velocity]')
        self.spin_action = np.clip(action, -1, 1)
        self.target_y = float(np.clip(self.target_y+.5*self.dt*self.spin_action[0], -.08, .08))
        self.target_height = float(np.clip(self.target_height+.5*self.dt*self.spin_action[1], .78, .89))
        for _ in range(10):
            self.reflex()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        angle, speed = self.pole_state()
        rotor = float(self.data.qpos[self.qids['pole_0_joint']])
        change = rotor-self.previous_rotor_angle
        self.previous_rotor_angle = rotor
        delta = self.data.site('pole_tip').xpos-self.data.body('pole_0').xpos
        local = self.data.body('pelvis').xmat.reshape(3, 3).T @ delta
        free = not self.pole_supported() and np.linalg.norm(local[1:]) > .18
        progress = self._rotation_progress(change, speed, free)
        self.contact_time += self.dt*(not free)
        z = float(self.data.body('pelvis').xpos[2])
        pitch, roll = (float(self.data.qpos[self.qids[name]]) for name in ('root_pitch', 'root_roll'))
        healthy = (z > .55 and abs(pitch) < .7 and abs(roll) < .6
                   and np.isfinite(self.data.qpos).all())
        reward = (8*progress/(2*math.pi)
                  - self.dt*(.01*(self.direction*speed-self.target_speed)**2
                             + .005*float(np.mean(self.data.act**2))+2*(not free)))
        if not healthy:
            reward -= 5.
        return self.observe(), float(reward), not healthy, self.steps*self.dt >= self.duration, self.info()

    def info(self):
        info = super().info()
        info.update(task='spin', spin_angle=self.pole_state()[0],
                    angular_velocity=self.pole_state()[1],
                    signed_turns=self.signed_angle/(2*math.pi),
                    consecutive_turns=self.streak_angle/(2*math.pi),
                    best_consecutive_turns=self.best_streak_angle/(2*math.pi),
                    completed_turns=int(max(0., self.signed_angle)/(2*math.pi)),
                    contact_time=self.contact_time,
                    pelvis_y=float(self.data.qpos[self.qids['root_y']]),
                    pelvis_roll=float(self.data.qpos[self.qids['root_roll']]))
        return info
