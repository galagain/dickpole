"""Manual viewer for the final muscle-driven environment."""
from queue import SimpleQueue
import time

import mujoco
import numpy as np

from .environment import MuscleHumanoid


def manual(seed=3000):
    import mujoco.viewer
    env = MuscleHumanoid(duration=3600.)
    env.reset(seed=seed)
    events = SimpleQueue()
    command = np.zeros(3)
    deadline = 0.
    ended = paused = False
    # GLFW: arrows translate/raise/lower; A/D tilt the pelvis.
    bindings = {263: (0, -.5), 262: (0, .5), 265: (2, .5),
                264: (2, -.5), 65: (1, -.2), 68: (1, .2)}
    print('←/→ translation | ↑/↓ hauteur | A/D inclinaison | Espace pause | R reset', flush=True)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.put) as viewer:
        viewer.cam.azimuth = 145
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
            viewer.cam.lookat[:] = [env.data.body('pelvis').xpos[0], 0, .85]
            viewer.sync()
            time.sleep(max(0, env.dt-(time.monotonic()-start)))
