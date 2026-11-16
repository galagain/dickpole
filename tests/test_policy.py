import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from dickpole.humanoid import HumanoidEnv
from dickpole.policy import TaskPolicy, model_hash

ROOT=Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_both_checkpoints_use_the_exported_common_model(self):
        import hashlib
        exported=hashlib.sha256((ROOT/'models/humanoid.xml').read_bytes()).hexdigest()
        self.assertEqual(exported,model_hash())
        for task in ('balance','spin'):
            policy=TaskPolicy(ROOT/'models'/task,task)
            self.assertEqual(policy.manifest['physical_model_sha256'],exported)
            action=policy.predict(HumanoidEnv(task=task))
            self.assertEqual(np.asarray(action).shape,(5,))
            self.assertTrue(np.isfinite(action).all())

    def test_identical_weights_and_inputs_have_identical_actions_for_both_rewards(self):
        from dickpole.controller import action,Stabilizer
        stabilizer=Stabilizer(ROOT/'models')
        with np.load(ROOT/'models/initial_profiles.npz') as bundle:
            parameters=bundle['parameters'][0].copy()
        for mix in (0.,.5,1.):
            a,b=HumanoidEnv(task='balance'),HumanoidEnv(task='spin')
            a.reset(seed=17);b.reset(seed=17)
            parameters[17]=mix
            for _ in range(40):
                x,y=action(a,parameters,stabilizer),action(b,parameters,stabilizer)
                np.testing.assert_array_equal(x,y)
                oa,ra,da,ta,ia=a.step(x);ob,rb,db,tb,ib=b.step(y)
                np.testing.assert_array_equal(oa,ob)
                self.assertEqual((da,ta,ia),(db,tb,ib))
            self.assertNotEqual(ra,rb)

    def test_reward_transfer_is_allowed_but_geometry_mismatch_is_rejected(self):
        transferred=TaskPolicy(ROOT/'models/balance','spin')
        self.assertEqual(transferred.task,'spin')
        self.assertEqual(transferred.trained_task,'balance')
        with tempfile.TemporaryDirectory() as directory:
            manifest=json.loads((ROOT/'models/spin/manifest.json').read_text())
            manifest['physical_model_sha256']='incompatible'
            Path(directory,'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'different physical model'):
                TaskPolicy(directory)

    def test_planning_does_not_advance_live_physics_or_controller(self):
        for task in ('balance','spin'):
            policy=TaskPolicy(ROOT/'models'/task)
            policy.parameters=policy.parameters[:1]
            env=HumanoidEnv(task=task,duration=.1)
            env.reset(seed=7)
            kind=mujoco.mjtState.mjSTATE_INTEGRATION
            before=np.empty(mujoco.mj_stateSize(env.model,kind));after=before.copy()
            mujoco.mj_getState(env.model,env.data,before,kind)
            observation=env.observe().copy()
            policy.select(env,horizon=.1)
            mujoco.mj_getState(env.model,env.data,after,kind)
            np.testing.assert_array_equal(before,after)
            np.testing.assert_array_equal(observation,env.observe())
            self.assertEqual(env.steps,0)
            self.assertEqual(env.control_phase,0)
            self.assertIsNone(env.lqr_anchor)

    def test_world_tilt_cannot_count_as_upright_balance(self):
        env=HumanoidEnv(task='balance')
        env.data.qpos[env.qids['root_roll']]=.35
        env.data.qpos[env.qids['pole_0_joint']]=np.pi
        mujoco.mj_forward(env.model,env.data)
        _,_,_,_,info=env.step([0,0,0,0,0])
        self.assertEqual(info['hold'],0.)


if __name__=='__main__':unittest.main()
