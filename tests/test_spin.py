import math
import unittest
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from dickpole.spin import SpinHumanoid


class SpinTests(unittest.TestCase):
    def test_native_muscles_without_rotor_motor(self):
        env = SpinHumanoid()
        self.assertEqual(env.model.nu, 24)
        self.assertEqual(env.model.nv, 27)
        self.assertTrue(np.all(env.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE))
        passive = {env.model.joint(n).id for n in
                   ('root_x', 'root_y', 'root_z', 'root_pitch', 'root_roll',
                    'pole_0_joint', 'pole_1_joint', 'pole_2_joint', 'pole_splay')}
        self.assertFalse(passive.intersection(map(int, env.model.wrap_objid)))
        self.assertAlmostEqual(env.pole_state()[0], 0.)
        np.testing.assert_array_equal(env.data.qvel, 0.)
        self.assertEqual(env.model.neq, 0)

    def test_stands_on_ground_without_external_forces(self):
        env = SpinHumanoid()
        forces = []
        for _ in range(300):
            _, _, done, _, info = env.step([0., 0.])
            self.assertFalse(done)
            forces.append(sum(info['ground_forces']))
        self.assertAlmostEqual(np.mean(forces[-100:])/(sum(env.model.body_mass)*9.81),
                               1., delta=.08)
        self.assertFalse(np.any(env.data.xfrc_applied))
        self.assertFalse(np.any(env.data.qfrc_applied))

    def test_rotation_is_relative_to_pelvis(self):
        env = SpinHumanoid()
        env.data.qpos[env.qids['root_roll']] = .3
        env.data.qvel[env.dofs['root_roll']] = 1.
        mujoco.mj_forward(env.model, env.data)
        angle, speed = env.pole_state()
        self.assertAlmostEqual(angle, 0.)
        self.assertAlmostEqual(speed, 0.)
        env.data.qvel[env.dofs['pole_0_joint']] = 2.
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual(env.pole_state()[1], 2.)

    def test_partial_swings_and_contacts_do_not_count_as_turns(self):
        env = SpinHumanoid()
        for _ in range(10):
            for delta in (.2,)*10+(-.2,)*10:
                env._rotation_progress(delta, delta/env.dt, True)
        self.assertAlmostEqual(env.signed_angle, 0.)
        self.assertLess(env.best_streak_angle, 2*math.pi)
        env._rotation_progress(.2, 10., False)
        self.assertEqual(env.streak_angle, 0.)
        for _ in range(40):
            env._rotation_progress(.2, 10., True)
        self.assertGreater(env.best_streak_angle, 2*math.pi)

    def test_front_rotor_clears_body_through_full_revolution(self):
        env = SpinHumanoid()
        for angle in np.linspace(0., 2*math.pi, 73):
            env.data.qpos[env.qids['pole_0_joint']] = angle
            mujoco.mj_forward(env.model, env.data)
            self.assertFalse(env.pole_supported())

    def test_gym_api(self):
        check_env(SpinHumanoid(), warn=True)

    def test_planning_preserves_live_world(self):
        from dickpole.spin_learning import SpinPolicy
        policy = SpinPolicy(Path(__file__).resolve().parents[1]/'models'/'spin')
        env = SpinHumanoid(duration=.1)
        env.reset(seed=3001)
        before = np.r_[env.data.qpos, env.data.qvel, env.data.act].copy()
        policy.select(env, horizon=.1)
        np.testing.assert_array_equal(before, np.r_[env.data.qpos, env.data.qvel, env.data.act])
        self.assertEqual(env.data.time, 0.)
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.signed_angle, 0.)

    def test_pretrained_rotations_from_rest(self):
        from dickpole.spin_learning import SpinPolicy
        policy = SpinPolicy(Path(__file__).resolve().parents[1]/'models'/'spin')
        env = SpinHumanoid()
        env.reset(seed=3001)
        for _ in range(1000):
            _, _, done, timeout, info = env.step(policy.predict(env))
            self.assertFalse(done)
        self.assertTrue(timeout)
        self.assertGreater(info['best_consecutive_turns'], 3.)


if __name__ == '__main__':
    unittest.main()
