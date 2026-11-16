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
            self.assertEqual(np.asarray(action).shape,(2,))
            self.assertTrue(np.isfinite(action).all())

    def test_incompatible_goal_and_geometry_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Expected spin'):
            TaskPolicy(ROOT/'models/balance','spin')
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
        _,_,_,_,info=env.step([0,0])
        self.assertEqual(info['hold'],0.)


if __name__=='__main__':unittest.main()
