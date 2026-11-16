<h1 align="center">DickPole</h1>
<p align="center"><strong>One ridiculous humanoid. Two control problems. Real physics.</strong></p>

<table>
<tr><th>Balance training preview</th><th>Fast rotation</th></tr>
<tr>
<td><a href="media/balance.mp4"><img src="media/balance.gif" width="384" alt="Balance-objective rollout, viewed from the front"></a></td>
<td><a href="media/spin.mp4"><img src="media/spin.gif" width="384" alt="Muscle-driven frontal helicopter rotations"></a></td>
</tr>
</table>

Swing a hanging appendage upright, or make it rotate quickly in front of the
body. Both tasks use **exactly the same 3D MuJoCo humanoid**, with 24 muscles,
articulated legs, passive arms and testicles, and frictional ground contact.
Only the pelvis motion targets can be commanded. The appendage has no motor.

The original rectangular pelvis and longer, three-segment **36 cm shaft** are
restored. A passive hinge at the base permits unlimited full revolutions, with
the same joint limits in both tasks. The anterior attachment provides rotational
clearance. A named camera on the anatomical front (+X) keeps the movement visible;
the small eyes also make the viewing direction unambiguous. Both demonstrations show the first ten seconds of a real rollout.

## Run the trained controllers

Use Python 3.13. Both checkpoints are included.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Swing up from below and balance — reproducible demonstration
.venv/bin/mjpython main.py play --task balance --seed 7

# Rotate quickly, using the same body and attachment
.venv/bin/mjpython main.py play --task spin --seed 7
```

On macOS, use `mjpython` for the viewer. Controller selection first evaluates
learned profiles in independent simulator copies. The live episode then starts
**hanging down at rest**. Playback repeats the requested seed.
Use `--no-plan` to run profile zero directly, without simulator lookahead.

| Task | Checkpoint | Status |
| --- | --- | --- |
| Balance | `models/balance` | Training checkpoint; upright balance is not yet solved on this geometry |
| Fast rotation | `models/spin` | **10/10** evaluated episodes; mean **2.46 turns/s** after four seconds |

The old model's hold times do not apply after restoring the geometry and
changing the control interface. The balance clip is a **training preview**,
not a successful upright demonstration (**0/4** successful balance episodes). See the current
[balance validation](reports/balance_validation.json) and
[rotation validation](reports/spin_validation.json). Measurements include
simulator-based profile selection.

## Train either goal

```bash
# Fit two independent policies on the common 3D environment
.venv/bin/python main.py train --task balance --steps 262144
.venv/bin/python main.py train --task spin --steps 262144

# Continue training from an included checkpoint
.venv/bin/python main.py train --task balance --pretrained --steps 262144
.venv/bin/python main.py train --task spin --pretrained --steps 262144

# Run your trained policies
.venv/bin/mjpython main.py play --task balance --model outputs/balance --seed 7
.venv/bin/mjpython main.py play --task spin --model outputs/spin --seed 7
```

Both tasks use **the same episodic cross-entropy policy search (CEM)** and
**the same 20-parameter controller architecture**. Training screens a common
initial policy pool, samples parameter variations and retains profiles with
the highest **sum of environment rewards**. There are no task-specific action
masks, search spaces, success bonuses or ranking rules in the trainer.

The shared controller can mix energy pumping and phase-feedback motion,
apply lateral pelvis-roll pumping, and engage a common local LQR controller.
**All these possibilities are available in both tasks.** Learned weights decide
which primitives to use; the controller never reads the task name.
This is calibrated parameter search, not a neural PPO policy learning muscle
coordination from scratch.

`--steps` is a simulation-decision budget, rounded up to complete the current
episode. `--population`, `--rounds`, `--seed` and `--output` control the search.
Offline LQR identification uses upright poses to linearize the plant. Training,
evaluation and playback episodes themselves start hanging down at rest.

## Explore and evaluate

```bash
.venv/bin/mjpython main.py env --task balance
.venv/bin/mjpython main.py env --task spin

# Evaluate the balance training checkpoint
.venv/bin/python main.py evaluate --task balance --seed 7 --episodes 1

# Test different initial states
.venv/bin/python main.py evaluate --task balance --seed 3000 --episodes 10
.venv/bin/python main.py evaluate --task spin --seed 3000 --episodes 10

# Export ten seconds of environment-only footage; requires FFmpeg
.venv/bin/python main.py video --task balance --seed 7
.venv/bin/python main.py video --task spin --seed 7
```

**Manual controls:** ← / → lateral · ↑ / ↓ height · W / S fore-aft ·
A / D pitch · Q / E roll · Space pause · R reset.
Evaluation reports are written to `outputs/<task>_evaluation.json`.

## Shared environment architecture

```python
from dickpole import HumanoidEnv

balance = HumanoidEnv(task="balance")
spin = HumanoidEnv(task="spin")
```

**Only the reward function changes with the task.** No task label or goal
indicator is passed to the policy.
**Geometry, masses, joint limits, muscles, contacts, initial-state distribution
and the five-action interface are shared.** Applying the same actions from the
same seed produces identical observations, physical trajectories, termination flags and
diagnostic information in both tasks. Their rewards differ.

- [humanoid.py](dickpole/humanoid.py): common 3D model, state, actions and task rewards.
- [environment.py](dickpole/environment.py): skeleton construction and muscle reflex.
- [controller.py](dickpole/controller.py): shared action-generating controller.
- [learning.py](dickpole/learning.py): shared return-maximizing CEM training loop.
- [policy.py](dickpole/policy.py): checkpoint validation, selection and evaluation.
- [calibration.py](dickpole/calibration.py): local LQR identification.
- [models/humanoid.xml](models/humanoid.xml): exported physical model.

Each checkpoint records the same physical-model SHA-256 fingerprint and its
trained objective. Checkpoints for another geometry or controller architecture
are rejected. Weights may be evaluated under the other reward: for example,
`play --task spin --model models/balance` uses balance-trained weights with the
spin objective, without changing the body or available actions.

## Physics and mathematical problem

The system has **27 degrees of freedom**: pelvis translation in three axes,
pelvis pitch and roll, articulated trunk and legs, passive arms, a frontal
rotor with out-of-plane bending, and two suspended testicular masses.
Pelvis yaw is constrained. Feet are not welded to the ground.

MuJoCo integrates at **500 Hz** using `implicitfast`; the controller acts at
**50 Hz**. The full constrained multibody dynamics have the form

```math
M(q)\ddot q + C(q,\dot q)+g(q)
= J_m(q)^T F_m + J_c(q)^T\lambda
- D\dot q-\nabla V_{\mathrm{passive}}(q).
```

Here $`M`$ is the inertia matrix, $`F_m`$ the muscle forces and $`\lambda`$ the
contact reactions. Ground friction and contact forces are solved by MuJoCo.
The body is supported by its feet and muscles; no external force holds it up.

The **24 native Hill-type muscle actuators** form antagonist pairs across the
trunk, hips, knees and ankles. Their force depends on activation, length and
shortening velocity. Activation has separate rise and fall time constants of
10 and 40 ms. Fixed tendon moment arms are 4 cm. A postural reflex converts
pelvis targets into joint torque requests and bounded muscle excitations.
Gravity and passive spring/damper joints drive the unactuated appendages.

The **five actions**, identical in both tasks, command pelvis target velocities:

| Action | Motion | Target bound | Maximum target rate |
| --- | --- | --- | --- |
| 0 | Lateral translation | ±0.14 m | 0.5 m/s |
| 1 | Height | 0.78–0.89 m | 0.5 m/s |
| 2 | Fore-aft translation | ±0.14 m | 0.6 m/s |
| 3 | Pitch | ±0.3 rad | 2.4 rad/s |
| 4 | Roll | ±0.3 rad | 2.4 rad/s |

Actual motion follows muscle dynamics and ground contact; an action never sets
the physical pelvis pose directly. The common **94-value observation** contains
joint positions and velocities, muscle activations, five targets, five previous
actions, rotor angle and speed, direction, a speed reference and time. There is
no task-dependent field.

For intuition, a rigid frontal pendulum with angle $`\theta=0`$ downward obeys

```math
I\ddot\theta+c\dot\theta+mgl\sin\theta
=-ml\left(\ddot y_p\cos\theta+\ddot z_p\sin\theta\right),
```

where $`l`$ is its centre-of-mass distance from the pivot. MuJoCo solves the
complete three-segment body rather than this single-rod approximation.
The controller must inject or remove pendulum energy through pivot motion.

### Balance objective

The reward encourages height and free upright time while penalizing speed,
contact, activation and falls. Let $`U`$ indicate a valid upright hold, $`A`$ be
mean squared activation, $`\alpha`$ the world-vertical error and $`\Omega`$ the
angular speed of the base-to-tip direction in world coordinates:

```math
r_t=\Delta t\left[2(1-\cos\theta)^2+100U
+4e^{-(\alpha/0.3)^2-(\Omega/2)^2}
-0.05\dot\theta^2-0.005A-2\mathbf{1}_{\mathrm{contact}}\right]
-100\mathbf{1}_{\mathrm{fall}}.
```

A valid hold requires the base-to-tip direction to lie within **0.21 rad of
world vertical**, angular speed below **1 rad/s**, and no appendage contact.
An episode succeeds with at least five continuous seconds of free balance and
no body fall throughout the 20-second episode. A fall means pelvis height below
0.55 m, pitch beyond 0.7 rad, or roll beyond 0.6 rad. The shared feedback primitive is intended to catch the upright phase; the
current checkpoint has not yet achieved a reliable swing-up and hold.

### Fast-rotation objective

Rotation is measured **relative to the pelvis**, using the unlimited rotor
coordinate to count signed turns. Rolling the body or repeatedly swinging
back and forth cannot accumulate free consecutive revolutions.
With $`A`$ the mean squared muscle activation and $`\omega_*=4\pi`$ rad/s,

```math
r_t=\frac{8\Delta\theta_{\mathrm{credited}}}{2\pi}
-\Delta t\left[0.01(\dot\theta-\omega_*)^2+0.005A
+2\mathbf{1}_{\mathrm{contact}}\right]
-100\mathbf{1}_{\mathrm{fall}}.
```

Positive advance earns credit only in free motion; reverse motion always
loses credit. Contact, reversal or sustained stalling resets the current
streak. A successful fast-rotation episode needs three consecutive free turns,
mean signed free speed of at least **1.5 turns/s after four seconds**, and no
body fall. The 2 turns/s reward target is soft, rather than a hard speed limit.
Search maximizes the raw accumulated reward, using the same rule in both tasks.

This is an approximate mechanical benchmark, not a validated anatomical model.
It uses rigid segments, generic muscle parameters and fixed tendon routing;
it does not simulate volumetric tissue, blood flow, fatigue or learned walking.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -q
```
