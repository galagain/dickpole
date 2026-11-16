import math
import unittest

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from dickpole.environment import MuscleHumanoid


class HumanoidTests(unittest.TestCase):
    def test_native_muscles_and_free_pelvis(self):
        env = MuscleHumanoid()
        self.assertEqual(env.model.nu, 14)
        self.assertTrue(np.all(env.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE))
        self.assertTrue(np.all(env.model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON))
        root_joints = [env.model.joint(name).id for name in ('root_x', 'root_z', 'root_pitch')]
        self.assertFalse(any(int(joint) in root_joints for joint in env.model.wrap_objid))
        self.assertEqual(env.model.neq, 0)  # No welded feet / support constraints.
        self.assertAlmostEqual(math.cos(env.pole_state()[0]), -1.)

    def test_stands_on_contacts_without_external_forces(self):
        env = MuscleHumanoid()
        forces = []
        for _ in range(300):
            _, _, done, _, info = env.step([0., 0., 0.])
            self.assertFalse(done)
            forces.append(sum(info['ground_forces']))
        weight = sum(env.model.body_mass)*9.81
        self.assertAlmostEqual(np.mean(forces[-100:])/weight, 1., delta=.08)
        self.assertTrue(np.all(env.data.xfrc_applied == 0))
        self.assertTrue(np.all(env.data.qfrc_applied == 0))

    def test_body_falls_without_muscles(self):
        env = MuscleHumanoid()
        for _ in range(1500):
            env.data.ctrl[:] = 0
            mujoco.mj_step(env.model, env.data)
        self.assertLess(env.data.body('pelvis').xpos[2], .55)

    def test_pelvic_tilt_and_lumbar_compensation(self):
        results = []
        for direction in (-1, 1):
            env = MuscleHumanoid()
            env.reset(seed=17)
            for step in range(100):
                _, _, done, _, info = env.step([0., direction/6 if step < 15 else 0., 0.])
                self.assertFalse(done)
            self.assertAlmostEqual(env.target_x, 0.)
            self.assertAlmostEqual(env.target_pitch, direction*.12)
            self.assertGreater(direction*info['pelvis_pitch'], .07)
            self.assertLess(direction*info['lumbar_angle'], -.07)
            self.assertTrue(np.all(env.data.qfrc_applied == 0))
            self.assertTrue(np.all(env.data.xfrc_applied == 0))
            results.append(info['pelvis_pitch'])
        self.assertGreater(results[1]-results[0], .2)

    def test_tilt_limits_and_action_validation(self):
        env = MuscleHumanoid()
        for _ in range(30):
            env.step([0., 1., 0.])
        self.assertAlmostEqual(env.target_pitch, .30)
        with self.assertRaises(ValueError):
            env.step([0.])
        with self.assertRaises(ValueError):
            env.step([0., float('nan'), 0.])

    def test_seed_and_gym_api(self):
        env = MuscleHumanoid()
        a, _ = env.reset(seed=17)
        b, _ = env.reset(seed=17)
        c, _ = env.reset(seed=18)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(b, c))
        check_env(env)

    def test_upright_is_not_inside_torso(self):
        env = MuscleHumanoid()
        env.data.qpos[env.qids['pole_0_joint']] = math.pi
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual(env.pole_state()[0], 0.)
        self.assertFalse(env.pole_supported())

    def test_measured_angular_speed_matches_tip_motion(self):
        env = MuscleHumanoid()
        env.data.qpos[env.qids['pole_0_joint']] = .5
        for i, value in enumerate((.0, .2, -.15)):
            joint = env.model.joint(f'pole_{i}_joint')
            if i:
                env.data.qpos[int(joint.qposadr[0])] = value
            env.data.qvel[int(joint.dofadr[0])] = i*.4+.3
        env.data.qvel[env.dofs['root_pitch']] = .2
        mujoco.mj_forward(env.model, env.data)
        angle, speed = env.pole_state()
        eps = 1e-7
        mujoco.mj_integratePos(env.model, env.data.qpos, env.data.qvel, eps)
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual((env.pole_state()[0]-angle)/eps, speed, places=5)

    def test_evaluation_always_starts_down_without_kick(self):
        env = MuscleHumanoid()
        for seed in range(10):
            env.reset(seed=seed)
            angle, speed = env.pole_state()
            self.assertAlmostEqual(math.cos(angle), -1.)
            self.assertEqual(speed, 0.)

    def test_physical_energy_and_no_rotor_inertia(self):
        env = MuscleHumanoid()
        self.assertTrue(np.all(env.model.dof_armature == 0.))
        self.assertAlmostEqual(env.pole_energy(), 0., places=10)
        env.data.qvel[0] = 1.
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual(env.pole_energy(), 0., places=10)
        env.data.qvel[:] = 0.
        env.data.qpos[env.qids['pole_0_joint']] = math.pi
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual(env.pole_energy(), 1., places=10)

    def test_residual_wrapper_and_teacher_do_not_teleport(self):
        from dickpole.rl_env import ResidualHumanoid
        from dickpole.controller import SwingUpTeacher
        env = MuscleHumanoid()
        teacher = SwingUpTeacher()
        positions, velocities = env.data.qpos.copy(), env.data.qvel.copy()
        teacher(env)
        np.testing.assert_array_equal(env.data.qpos, positions)
        np.testing.assert_array_equal(env.data.qvel, velocities)
        check_env(ResidualHumanoid())

    def test_stabilizer_ignores_residual_after_capture(self):
        from dickpole.rl_env import ResidualHumanoid
        a, b = ResidualHumanoid(), ResidualHumanoid()
        for env in (a, b):
            env.reset(seed=7)
            env.unwrapped.data.qpos[env.unwrapped.qids['pole_0_joint']] = math.pi
            env.unwrapped.balance_mode = True
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        _, _, _, _, info_a = a.step([1., -1., .5])
        _, _, _, _, info_b = b.step([0., 0., 0.])
        self.assertEqual(info_a['residual_scale'], 0.)
        self.assertEqual(info_a['applied_action'], info_b['applied_action'])
        np.testing.assert_array_equal(a.unwrapped.data.qpos, b.unwrapped.data.qpos)


if __name__ == '__main__':
    unittest.main()
