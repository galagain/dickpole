"""Render the first ten seconds of the environment, without any overlays."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco

from .humanoid import HumanoidEnv


def render_presentation(policy, output, seed=3000, plan=True):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg is required to export the video')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = HumanoidEnv(task=policy.task, direction=policy.direction)
    env.reset(seed=seed)
    if plan:
        policy.select(env)
    sim = env
    width, height, fps = 1280, 720, 25
    sim.model.vis.global_.offwidth = max(width, sim.model.vis.global_.offwidth)
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 12, -12, 2.8
    command = [executable, '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    with tempfile.TemporaryFile() as error:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error)
        try:
            for frame in range(10*fps):
                while sim.data.time+1e-8 < frame/fps:
                    action = policy.predict(env)
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        raise RuntimeError('The recorded episode fell')
                camera.lookat[:] = [*sim.data.body('pelvis').xpos[:2], .85]
                renderer.update_scene(sim.data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
            renderer.close()
            env.close()
        if code:
            error.seek(0)
            raise RuntimeError(error.read().decode(errors='replace'))
    # Animated Markdown preview works in README renderers that strip video tags.
    preview = output.with_suffix('.gif')
    subprocess.run([executable, '-y', '-loglevel', 'error', '-i', str(output),
                    '-filter_complex',
                    '[0:v]fps=15,scale=768:-1:flags=lanczos,split[a][b];'
                    '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                    '-loop', '0', str(preview)], check=True)
    print(f'Video ready: {output} (10 s, environment only)', flush=True)
