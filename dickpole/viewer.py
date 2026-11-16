"""Manual viewer for the final muscle-driven environment."""
from queue import SimpleQueue
import time

import mujoco
import numpy as np

from .humanoid import HumanoidEnv


def manual(seed=3000, task='balance'):
    import mujoco.viewer
    env = HumanoidEnv(task=task, duration=3600.)
    env.reset(seed=seed)
    events = SimpleQueue()
    command = np.zeros(env.action_space.shape)
    deadline = 0.
    ended = paused = False
    # GLFW: arrows translate/raise/lower; A/D tilt the pelvis.
    bindings = {263: (0, -.5), 262: (0, .5), 265: (1, .5), 264: (1, -.5)}
    print('←/→ lateral motion | ↑/↓ height | Space pause | R reset', flush=True)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.put) as viewer:
        viewer.cam.azimuth = 12
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.
        while viewer.is_running():
            start = time.monotonic()
            while not events.empty():
                key = events.get()
                if key == 82:
                    env.reset(seed=seed)
                    ended = False
                    command[:] = 0.
                elif key == 32:
                    paused = not paused
                elif key in bindings:
                    axis, value = bindings[key]
                    command[:] = 0.
                    command[axis] = value
                    deadline = time.monotonic()+.18
            if time.monotonic() > deadline:
                command[:] = 0.
            if not paused and not ended:
                _, _, ended, _, _ = env.step(command)
                if ended:
                    print('Chute : R pour recommencer', flush=True)
            viewer.cam.lookat[:] = [*env.data.body('pelvis').xpos[:2], .85]
            viewer.sync()
            time.sleep(max(0, env.dt-(time.monotonic()-start)))


def play(policy, seed=3000, plan=True):
    import mujoco.viewer
    env = HumanoidEnv(task=policy.task, direction=policy.direction)
    env.reset(seed=seed)
    if plan:
        print('Selecting a controller in independent simulations...', flush=True)
        policy.select(env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 12, -10, 2.8
        while viewer.is_running():
            start = time.monotonic()
            _, _, done, timeout, info = env.step(policy.predict(env))
            viewer.cam.lookat[:] = [*env.data.body('pelvis').xpos[:2], .85]
            viewer.sync()
            if done or timeout:
                print(f"Hold: {info['best_hold']:.2f}s | speed: {info['mean_free_rps']:.2f} rps | fell: {done}", flush=True)
                env.reset(seed=seed)
                if plan:
                    policy.select(env)
            time.sleep(max(0.,env.dt-(time.monotonic()-start)))
    env.close()
