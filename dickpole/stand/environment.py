"""Sagittal musculoskeletal DickPole. All propulsion comes from muscles/contact."""
import math
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


JOINTS = ('lumbar', 'hip_l', 'knee_l', 'ankle_l', 'hip_r', 'knee_r', 'ankle_r')


def model_xml():
    root = ET.Element('mujoco', model='DickPole musculoskeletal')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast', iterations='60', gravity='0 0 -9.81')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1200', offheight='800')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    defaults = ET.SubElement(root, 'default')
    # Muscle-driven anatomy has no motor rotor attached to its passive joints.
    ET.SubElement(defaults, 'joint', damping='.5', armature='0')
    ET.SubElement(defaults, 'geom', friction='1.0 .01 .001', solref='.008 1', solimp='.95 .99 .001', rgba='.8 .6 .48 1', contype='1', conaffinity='2')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -3 5', dir='0 0 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='8 4 .1', rgba='.18 .23 .28 1', contype='2', conaffinity='7')
    ET.SubElement(world, 'camera', name='side', pos='2 -4 1.8', xyaxes='1 .5 0 -.1 .2 1')
    pelvis = ET.SubElement(world, 'body', name='pelvis', pos='0 0 .85')
    for name, kind, axis in [('root_x', 'slide', '1 0 0'), ('root_z', 'slide', '0 0 1'), ('root_pitch', 'hinge', '0 1 0')]:
        ET.SubElement(pelvis, 'joint', name=name, type=kind, axis=axis, damping='0', armature='0', limited='false')
    ET.SubElement(pelvis, 'geom', name='pelvis_geom', type='box', size='.10 .14 .09', mass='10', rgba='.3 .45 .7 1')
    torso = ET.SubElement(pelvis, 'body', name='torso', pos='0 0 .09')
    ET.SubElement(torso, 'joint', name='lumbar', axis='0 1 0', range='-.65 .65', stiffness='10')
    ET.SubElement(torso, 'geom', type='capsule', fromto='0 0 0 0 0 .43', size='.12', mass='25', rgba='.2 .65 .6 1')
    ET.SubElement(torso, 'geom', name='head', type='sphere', pos='0 0 .61', size='.11', mass='5')
    for side, y in [('l', -.11), ('r', .11)]:
        # Passive shoulders, upper/lower arms form a compliant two-link chain.
        arm = ET.SubElement(torso, 'body', name=f'arm_{side}', pos=f'0 {y*1.8} .37')
        ET.SubElement(arm, 'joint', name=f'shoulder_{side}', axis='0 1 0', range='-1.2 1.2', stiffness='3', damping='.8')
        ET.SubElement(arm, 'geom', type='capsule', fromto='0 0 0 0 0 -.27', size='.035', mass='1.7')
        forearm = ET.SubElement(arm, 'body', pos='0 0 -.27')
        ET.SubElement(forearm, 'joint', name=f'elbow_{side}', axis='0 1 0', range='-1.8 .1', stiffness='2', damping='.5')
        ET.SubElement(forearm, 'geom', type='capsule', fromto='0 0 0 0 0 -.25', size='.03', mass='1.2')
        thigh = ET.SubElement(pelvis, 'body', name=f'thigh_{side}', pos=f'0 {y} 0')
        ET.SubElement(thigh, 'joint', name=f'hip_{side}', axis='0 1 0', range='-1.6 1.0')
        ET.SubElement(thigh, 'geom', type='capsule', fromto='0 0 0 0 0 -.4', size='.055', mass='7')
        shin = ET.SubElement(thigh, 'body', name=f'shin_{side}', pos='0 0 -.4')
        ET.SubElement(shin, 'joint', name=f'knee_{side}', axis='0 1 0', range='0 2.2')
        ET.SubElement(shin, 'geom', type='capsule', fromto='0 0 0 0 0 -.4', size='.045', mass='3.5')
        foot = ET.SubElement(shin, 'body', name=f'foot_{side}', pos='0 0 -.4')
        ET.SubElement(foot, 'joint', name=f'ankle_{side}', axis='0 1 0', range='-.8 .8')
        ET.SubElement(foot, 'geom', name=f'foot_{side}_geom', type='box', pos='.035 0 -.06', size='.15 .06 .04', mass='1')
        ET.SubElement(foot, 'site', name=f'sole_{side}', pos='0 0 -.10', size='.012', rgba='1 .8 0 1')
    # Downward chain with flexible base and two passive bending joints.
    parent = pelvis
    for i in range(3):
        segment = ET.SubElement(parent, 'body', name=f'pole_{i}', pos='.17 0 -.02' if i == 0 else '0 0 -.12')
        ET.SubElement(segment, 'joint', name=f'pole_{i}_joint', axis='0 -1 0', limited='false',
                      damping='.0008' if i == 0 else '.025', stiffness='0' if i == 0 else '3')
        ET.SubElement(segment, 'geom', name=f'pole_{i}_geom', type='capsule', fromto='0 0 0 0 0 -.12', size='.022', mass='.075', rgba='.93 .67 .57 1', contype='4', conaffinity='7')
        if i == 2:
            ET.SubElement(segment, 'site', name='pole_tip', pos='0 0 -.12', size='.01')
        parent = segment
    for side, y in [('l', -.045), ('r', .045)]:
        ball = ET.SubElement(pelvis, 'body', name=f'testis_{side}', pos=f'.10 {y} -.04')
        ET.SubElement(ball, 'joint', name=f'testis_{side}_joint', axis='0 1 0', limited='false', stiffness='.015', damping='.003')
        ET.SubElement(ball, 'geom', type='capsule', fromto='0 0 0 0 0 -.09', size='.005', mass='.005')
        ET.SubElement(ball, 'geom', type='ellipsoid', pos='0 0 -.11', size='.028 .025 .035', mass='.035')
    # Native Hill-type muscle actuators pull against fixed tendon moment arms.
    tendons = ET.SubElement(root, 'tendon')
    actuators = ET.SubElement(root, 'actuator')
    for joint in JOINTS:
        for direction, sign in [('flexor', -1), ('extensor', 1)]:
            name = f'{joint}_{direction}'
            tendon = ET.SubElement(tendons, 'fixed', name=name)
            ET.SubElement(tendon, 'joint', joint=joint, coef=str(sign*.04))
            ET.SubElement(actuators, 'muscle', name=name, tendon=name, force='6000',
                          lengthrange='-.12 .12', range='.75 1.05', timeconst='.01 .04')
    # Avoid neighbouring-body overlap; all limbs still collide with the floor.
    contact = ET.SubElement(root, 'contact')
    for a, b in [('pelvis', 'pole_0'), ('pelvis', 'testis_l'), ('pelvis', 'testis_r'),
                 ('pole_0', 'testis_l'), ('pole_0', 'testis_r'), ('pole_1', 'testis_l'), ('pole_1', 'testis_r')]:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    return ET.tostring(root, encoding='unicode')


class MuscleHumanoid(gym.Env):
    """Pelvis translation, tilt and height commands with a muscle reflex."""
    metadata = {'render_modes': []}
    dt = .02

    def __init__(self, duration=20.):
        self.model = mujoco.MjModel.from_xml_string(model_xml())
        self.data = mujoco.MjData(self.model)
        self.duration = duration
        self.tilt_rate = 2.4
        self._tip_jac = np.zeros((3, self.model.nv))
        self._base_jac = np.zeros((3, self.model.nv))
        self._com_jac = np.zeros((3, self.model.nv))
        self._rotation_jac = np.zeros((3, self.model.nv))
        self.qids = {name: int(self.model.jnt_qposadr[self.model.joint(name).id])
                     for name in JOINTS + ('root_x', 'root_z', 'root_pitch', 'pole_0_joint')}
        self.dofs = {name: int(self.model.jnt_dofadr[self.model.joint(name).id]) for name in self.qids}
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, self.observe().shape, dtype=np.float32)

    def observe(self):
        q, v = self.data.qpos.copy(), self.data.qvel
        q[0] -= self.support_origin
        angle, speed = self.pole_state()
        extra = [self.target_height, self.last_height_action]
        return np.concatenate((q, v*.2, self.data.act,
                               [self.target_x, self.last_action, self.data.time/20.,
                                math.sin(angle), math.cos(angle), speed*.1,
                                self.target_pitch, self.last_pitch_action,
                                float(self.balance_mode), min(5., self.pole_energy())], extra)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Crouched stance; pole starts exactly hanging down.
        for side in ('l', 'r'):
            self.data.qpos[self.qids[f'hip_{side}']] = -.35
            self.data.qpos[self.qids[f'knee_{side}']] = .70
            self.data.qpos[self.qids[f'ankle_{side}']] = -.35
        self.data.qpos[self.qids['root_z']] = .1 + .8*math.cos(.35) - .85
        for side in ('l', 'r'):
            elbow = self.model.joint(f'elbow_{side}')
            self.data.qpos[int(elbow.qposadr[0])] = self.np_random.uniform(-.015, 0.)
            self.model.geom_friction[self.model.geom(f'foot_{side}_geom').id, 0] = self.np_random.uniform(.9, 1.1)
        self.target_x = self.last_action = self.best_hold = self.hold = self.upright_time = 0.
        self.target_pitch = self.last_pitch_action = 0.
        self.support_origin = 0.
        self.balance_mode = False
        self.target_height = .1+.8*math.cos(.35)
        self.last_height_action = 0.
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), self.info()

    def posture_targets(self):
        q = self.data.qpos
        pitch = q[self.qids['root_pitch']]
        # Analytic inverse kinematics for two equal leg segments, feet at x=0.
        dx = -self.target_x
        dz = -(self.target_height-.1)
        distance = min(.795, max(.45, math.hypot(dx, dz)))
        knee = 2*math.acos(distance/.8)
        hip_global = math.atan2(-dx, -dz) - knee/2
        balance = (pitch-self.target_pitch + .2*self.data.qvel[self.dofs['root_pitch']]
                   + (q[self.qids['root_x']]-self.support_origin-self.target_x)
                   + .8*self.data.qvel[self.dofs['root_x']])
        ankle = -hip_global-knee+balance
        # Hip muscles tilt the pelvis relative to the supporting legs; lumbar
        # muscles counter-rotate the torso. No force/torque is applied to root.
        hip = hip_global-self.target_pitch
        lumbar = -.3*pitch-.7*self.target_pitch
        return [lumbar, hip, knee, ankle, hip, knee, ankle]

    def reflex(self):
        q, v, m, d = self.data.qpos, self.data.qvel, self.model, self.data
        targets = self.posture_targets()
        kp = [450, 450, 550, 350, 450, 550, 350]
        kd = [40, 35, 35, 25, 35, 35, 25]
        for j, name in enumerate(JOINTS):
            error = targets[j]-q[self.qids[name]]
            torque = kp[j]*error - kd[j]*v[self.dofs[name]]
            # Muscle shortening produces negative force along each tendon.
            for k, sign in ((2*j, -1), (2*j+1, 1)):
                gain = mujoco.mju_muscleGain(d.actuator_length[k], d.actuator_velocity[k],
                                             m.actuator_lengthrange[k], m.actuator_acc0[k], m.actuator_gainprm[k, :9])
                capacity = max(1., abs(gain*.04))
                d.ctrl[k] = min(1., .015 + max(0., -sign*torque)/capacity)

    def pole_state(self):
        base = self.data.body('pole_0').xpos
        tip = self.data.site('pole_tip').xpos
        direction = tip-base
        angle = math.atan2(direction[0], direction[2])
        # Angular speed of the measured base-to-tip direction. Summing hinge
        # speeds instead measures the terminal segment and is wrong when bent.
        mujoco.mj_jacSite(self.model, self.data, self._tip_jac, None,
                          self.model.site('pole_tip').id)
        mujoco.mj_jacBody(self.model, self.data, self._base_jac, None,
                          self.model.body('pole_0').id)
        velocity = (self._tip_jac-self._base_jac) @ self.data.qvel
        speed = ((direction[2]*velocity[0]-direction[0]*velocity[2]) /
                 max(1e-10, direction[0]**2+direction[2]**2))
        return angle, speed

    def pole_supported(self):
        """Touching torso/limbs/floor cannot count as free upright balance."""
        pole_geoms = {self.model.geom(f'pole_{i}_geom').id for i in range(3)}
        return any((c.geom1 in pole_geoms) != (c.geom2 in pole_geoms)
                   for c in self.data.contact if c.dist <= 0)

    def pole_energy(self):
        """Exact rigid-segment energy relative to the translating base.

        Zero is straight down at rest; one is straight up at rest. Includes
        bending energy, segment inertia and COM translation, not a rod proxy.
        """
        m, d = self.model, self.data
        base = d.body('pole_0').xpos
        mujoco.mj_jacBody(m, d, self._base_jac, None, m.body('pole_0').id)
        base_velocity = self._base_jac @ d.qvel
        energy = minimum = 0.
        for i in range(3):
            body = m.body(f'pole_{i}').id
            mass = m.body_mass[body]
            mujoco.mj_jacBodyCom(m, d, self._com_jac, self._rotation_jac, body)
            velocity = self._com_jac @ d.qvel-base_velocity
            omega = d.ximat[body].reshape(3, 3).T @ (self._rotation_jac @ d.qvel)
            energy += .5*(mass*float(velocity @ velocity)
                          + float(m.body_inertia[body] @ (omega*omega)))
            energy += mass*9.81*(d.xipos[body, 2]-base[2])
            minimum += mass*9.81*(i*.12+.06)
            if i:
                joint = m.joint(f'pole_{i}_joint').id
                energy += .5*m.jnt_stiffness[joint]*d.qpos[m.jnt_qposadr[joint]]**2
        return (energy+minimum)/(2*minimum)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape:
            raise ValueError(f'Expected {self.action_space.shape[0]} pelvis commands')
        if not np.isfinite(action).all():
            raise ValueError('Action must be finite')
        clipped = np.clip(action, -1, 1)
        translation, tilt = clipped[:2]
        self.last_action = float(translation)
        self.last_pitch_action = float(tilt)
        # A desired pelvis velocity integrates to a bounded stance target.
        self.target_x = float(np.clip(self.target_x + .012*translation, -.14, .14))
        # Anteversion/retroversion: ~17 degree target bound; actual motion is
        # produced by the force/velocity/activation-limited muscles.
        self.target_pitch = float(np.clip(self.target_pitch + self.tilt_rate*self.dt*tilt, -.30, .30))
        self.last_height_action = float(clipped[2])
        self.target_height = float(np.clip(self.target_height+.5*self.dt*clipped[2], .70, .87))
        for _ in range(10):
            self.reflex()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        angle, speed = self.pole_state()
        # Choose a fixed stance frame at capture. This changes the controller's
        # position reference only, never qpos, qvel or any applied force.
        if not self.balance_mode and abs(angle) < .25 and abs(speed) < 2.5 and not self.pole_supported():
            self.support_origin = float((self.data.site('sole_l').xpos[0]
                                         +self.data.site('sole_r').xpos[0])/2)
            self.balance_mode = True
        elif self.balance_mode and abs(angle) > .7:
            self.balance_mode = False
        z = self.data.body('pelvis').xpos[2]
        pitch = self.data.qpos[self.qids['root_pitch']]
        upright = abs(angle) < .21 and abs(speed) < 1. and not self.pole_supported()
        self.hold = self.hold+self.dt if upright else 0.
        self.best_hold = max(self.best_hold, self.hold)
        self.upright_time += self.dt*upright
        healthy = z > .55 and abs(pitch) < .7 and np.isfinite(self.data.qpos).all()
        energy = self.pole_energy()
        height = (1+math.cos(angle))/2
        reward = (2*height*height + 6*upright + .3*(1-abs(1-energy))
                  - .005*float(np.mean(self.data.act**2)) - .03*pitch*pitch)
        terminated = not healthy
        if terminated:
            reward -= 2
        return self.observe(), float(reward), terminated, self.steps*self.dt >= self.duration, self.info()

    def info(self):
        forces = [0., 0.]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, side in enumerate(('l', 'r')):
                foot = self.model.geom(f'foot_{side}_geom').id
                floor = self.model.geom('floor').id
                if {c.geom1, c.geom2} == {foot, floor}:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    forces[j] += wrench[0]
        return dict(time=float(self.data.time), best_hold=self.best_hold,
                    upright_time=self.upright_time, hold=self.hold,
                    pelvis_height=float(self.data.body('pelvis').xpos[2]),
                    pelvis_pitch=float(self.data.qpos[self.qids['root_pitch']]),
                    pelvis_target_pitch=self.target_pitch,
                    lumbar_angle=float(self.data.qpos[self.qids['lumbar']]),
                    pelvis_target_height=self.target_height,
                    pole_supported=self.pole_supported(),
                    balance_mode=self.balance_mode,
                    ground_forces=forces)
