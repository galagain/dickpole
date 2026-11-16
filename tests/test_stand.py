import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from dickpole.stand.policy import PolicyLibrary
from dickpole.stand.environment import MuscleHumanoid
from dickpole.stand.rl_env import ResidualHumanoid
from dickpole.stand.controller import SwingUpTeacher


class ZeroResidualLibrary(PolicyLibrary):
    """Exercise the planner without needing a downloaded/trained PPO artifact."""
    def __init__(self):
        self.profiles = np.zeros((1, 6))
        self.selected = None

    def apply(self, index):
        self.selected = index

    def predict(self, observation):
        return np.zeros(3, dtype=np.float32)


class StandPlanningTests(unittest.TestCase):
    def test_restored_model_and_assets_match_the_manifest(self):
        from dickpole.stand.environment import model_xml
        folder=Path(__file__).resolve().parents[1]/'models/stand'
        manifest=json.loads((folder/'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(model_xml().encode()).hexdigest(),manifest['physics_sha256'])
        for filename,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder/filename).read_bytes()).hexdigest(),digest)

    def test_cli_stand_and_balance_select_original_pipeline(self):
        from main import main
        for task in ('stand','balance'):
            with patch('sys.argv',['main.py','play','--task',task]), patch('dickpole.stand.cli.run') as run:
                main()
                self.assertEqual(run.call_args.args[0].task,task)
                self.assertEqual(run.call_args.args[0].seed,3000)

    def test_cli_spin_keeps_current_checkpoint_loader(self):
        from main import main
        with patch('sys.argv',['main.py','play','--task','spin','--no-plan']), \
             patch('dickpole.policy.load_policy') as load, \
             patch('dickpole.viewer.play') as play:
            main()
            self.assertEqual(load.call_args.args[0].parts[-2:],('models','spin'))
            self.assertEqual(load.call_args.args[1],'spin')
            self.assertFalse(play.call_args.kwargs['plan'])

    def test_episode_horizon_does_not_change_observation_clock(self):
        a, b = MuscleHumanoid(duration=20), MuscleHumanoid(duration=40)
        a.reset(seed=19)
        b.reset(seed=19)
        for _ in range(50):
            a.step([0., 0., 0.])
            b.step([0., 0., 0.])
        np.testing.assert_array_equal(a.observe(), b.observe())

    def test_actual_down_to_free_balance_with_muscles_only(self):
        env = MuscleHumanoid(duration=20)
        env.reset(seed=7)
        teacher = SwingUpTeacher()
        for _ in range(1000):
            _, _, done, _, info = env.step(teacher(env))
            self.assertFalse(done)
        self.assertGreater(info['best_hold'], 10.)
        self.assertFalse(info['pole_supported'])
        self.assertTrue(np.all(env.data.qfrc_applied == 0))
        self.assertTrue(np.all(env.data.xfrc_applied == 0))

    def test_planner_isolates_live_state_and_predicts_real_rollout(self):
        live = ResidualHumanoid(duration=40)
        live.reset(seed=7)
        sim = live.unwrapped
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(sim.model, spec))
        after = before.copy()
        mujoco.mj_getState(sim.model, sim.data, before, spec)
        observation = live.observe().copy()
        plan = ZeroResidualLibrary().select(live, horizon=12.)
        mujoco.mj_getState(sim.model, sim.data, after, spec)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(observation, live.observe())
        self.assertEqual(sim.data.time, 0.)
        self.assertTrue(plan['predicted_success'])
        for _ in range(600):
            _, _, done, _, info = live.step(np.zeros(3))
            self.assertFalse(done)
        self.assertAlmostEqual(info['best_hold'], plan['attempts'][0]['hold'], places=10)


if __name__ == '__main__':
    unittest.main()
