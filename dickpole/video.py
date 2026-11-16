"""Render the first ten seconds of the environment, without any overlays."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco

from .policy import PolicyLibrary
from .rl_env import ResidualHumanoid


def render_presentation(model, output, seed=3000, task='balance'):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg is required to export the video')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if task == 'spin':
        from .spin import SpinHumanoid
        from .spin_learning import SpinPolicy
        library = SpinPolicy(model)
        env = SpinHumanoid(duration=20., direction=library.direction)
    else:
        library = PolicyLibrary(model)
        env = ResidualHumanoid(duration=40.)
    obs, _ = env.reset(seed=seed)
    library.select(env)
    sim = env.unwrapped
    width, height, fps = 1280, 720, 25
    sim.model.vis.global_.offwidth = max(width, sim.model.vis.global_.offwidth)
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 110, -12, 2.8
    if task == 'spin':
        camera.azimuth = 10
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
                    action = library.predict(env) if task == 'spin' else library.predict(obs)
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
