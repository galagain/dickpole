import math
import unittest

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from dickpole.humanoid import HumanoidEnv, humanoid_model_xml


class HumanoidTests(unittest.TestCase):
    def test_both_tasks_have_identical_physics(self):
        a, b = HumanoidEnv(task='balance'), HumanoidEnv(task='spin')
        a.reset(seed=17); b.reset(seed=17)
        for action in ([.1,-.1],[-.1,.1],[0.,0.])*30:
            a.step(action); b.step(action)
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
        np.testing.assert_array_equal(a.data.act,b.data.act)
        self.assertEqual(a.model.nv,27)
        self.assertEqual(a.model.nu,24)
        self.assertEqual(a.action_space.shape,b.action_space.shape)
        self.assertEqual(a.observation_space.shape,b.observation_space.shape)
        self.assertEqual(a.observe()[-1],1.)
        self.assertEqual(b.observe()[-1],0.)

    def test_attachment_meets_body_surface(self):
        env=HumanoidEnv()
        pelvis_front=float(env.model.geom('pelvis_geom').size[0])
        attachment=float(env.model.body('pole_0').pos[0])
        root_radius=float(env.model.geom('anatomical_root').size[0])
        self.assertLess(attachment-pelvis_front,root_radius)
        self.assertAlmostEqual(np.linalg.norm(env.data.site('pole_tip').xpos-env.data.body('pole_0').xpos),.21)
        self.assertNotIn('rotor_attachment',humanoid_model_xml())

    def test_only_native_muscles_actuate_the_skeleton(self):
        env=HumanoidEnv()
        self.assertTrue(np.all(env.model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE))
        self.assertTrue(np.all(env.model.actuator_trntype==mujoco.mjtTrn.mjTRN_TENDON))
        passive={env.model.joint(n).id for n in ('root_x','root_y','root_z','root_pitch',
                                                'root_roll','pole_0_joint','pole_splay',
                                                'pole_1_joint','pole_2_joint')}
        self.assertFalse(passive.intersection(map(int,env.model.wrap_objid)))
        self.assertEqual(env.model.neq,0)
        self.assertAlmostEqual(env.pole_state()[0],0.)
        np.testing.assert_array_equal(env.data.qvel,0.)

    def test_ground_support_without_external_forces(self):
        env=HumanoidEnv();env.reset(seed=7);forces=[]
        for _ in range(300):
            _,_,done,_,info=env.step([0,0]);self.assertFalse(done)
            forces.append(sum(info['ground_forces']))
        self.assertAlmostEqual(np.mean(forces[-100:])/(sum(env.model.body_mass)*9.81),1.,delta=.08)
        self.assertFalse(np.any(env.data.xfrc_applied))
        self.assertFalse(np.any(env.data.qfrc_applied))

    def test_rotation_is_relative_to_pelvis(self):
        env=HumanoidEnv()
        env.data.qpos[env.qids['root_roll']]=.3
        env.data.qvel[env.dofs['root_roll']]=1.
        mujoco.mj_forward(env.model,env.data)
        angle,speed=env.pole_state()
        self.assertAlmostEqual(angle,0.)
        self.assertAlmostEqual(speed,0.)

    def test_oscillations_cannot_accumulate_turns(self):
        env=HumanoidEnv()
        for _ in range(10):
            for delta in (.2,)*10+(-.2,)*10:
                env._rotation_progress(delta,delta/env.dt,True)
        self.assertAlmostEqual(env.signed_angle,0.)
        self.assertLess(env.best_streak_angle,2*math.pi)
        env._rotation_progress(.2,10.,False)
        self.assertEqual(env.streak_angle,0.)

    def test_seeded_reset_repeats_the_trajectory(self):
        env=HumanoidEnv()
        states=[]
        for _ in range(2):
            env.reset(seed=7)
            for _ in range(100):env.step([.1,-.1])
            states.append(env.data.qpos.copy())
        np.testing.assert_array_equal(*states)

    def test_gym_contract_for_both_goals(self):
        for task in ('balance','spin'):
            check_env(HumanoidEnv(task=task),skip_render_check=True)


if __name__=='__main__':unittest.main()
