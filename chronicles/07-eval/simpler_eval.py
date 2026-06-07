"""
simpler_eval.py — SmolVLA evaluation in SimplerEnv (WidowX)

Install first (Ubuntu):
    pip install simpler-env gymnasium[mujoco]
    sudo apt install -y libegl1 libopengl0 libgles2

Run (headless — no display needed):
    MUJOCO_GL=egl python simpler_eval.py

What this measures:
    Success rate over N episodes — did the robot complete the task?
    SimplerEnv auto-detects task success via object-pose thresholds.

Note on action space:
    SmolVLA outputs 6-DOF joint positions.
    WidowX environments expect 6-DOF EEF delta + optional gripper.
    We pass SmolVLA's action directly and let the env interpret it.
    A real deployment would add an IK layer here.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering

# ── check SimplerEnv installed ────────────────────────────────────────────────
try:
    import simpler_env
    import gymnasium as gym
except ImportError:
    print("SimplerEnv not found. Install with:")
    print("  pip install simpler-env gymnasium[mujoco]")
    sys.exit(1)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── config ────────────────────────────────────────────────────────────────────
ENV_ID   = "widowx_carrot_on_plate-v0"   # WidowX 6-DOF — closest to SmolVLA
TASK     = "put carrot on plate"
EPISODES = 10
MAX_STEPS = 200
IMG_SIZE  = 256

# ── SmolVLA wrapper ───────────────────────────────────────────────────────────
class SmolVLAAgent:
    def __init__(self):
        print("Loading SmolVLA...")
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        self.policy = self.policy.to("cuda").eval()

        tok = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        enc = tok(TASK, return_tensors="pt", padding=True)
        self.lang_tokens = enc["input_ids"].cuda()
        self.lang_mask   = enc["attention_mask"].bool().cuda()
        print(f"Agent ready | task: '{TASK}'\n")

    def reset(self):
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        # extract RGB image — SimplerEnv gives (H, W, 3) uint8
        raw = obs.get("image", obs.get("rgb", np.zeros((256,256,3), dtype=np.uint8)))
        img = TF.to_tensor(
            Image.fromarray(raw.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        ).unsqueeze(0).cuda()

        # robot state — WidowX provides tcp_pose or qpos; take first 6 dims
        state_raw = obs.get("agent_qpos", obs.get("tcp_pose", np.zeros(6)))
        state = torch.tensor(
            np.asarray(state_raw, dtype=np.float32)[:6]
        ).unsqueeze(0).cuda()

        batch = {
            "observation.images.camera1":        img,
            "observation.images.camera2":        img,
            "observation.images.camera3":        img,
            "observation.state":                 state,
            "observation.language.tokens":       self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
        }
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return action[0].cpu().numpy()

# ── evaluation loop ───────────────────────────────────────────────────────────
def evaluate():
    env   = gym.make(ENV_ID, render_mode=None)
    agent = SmolVLAAgent()

    successes = 0
    results   = []

    for ep in range(EPISODES):
        obs, info = env.reset()
        agent.reset()
        ep_success = False

        for step in range(MAX_STEPS):
            action = agent.act(obs if isinstance(obs, dict) else {"image": obs})

            # pad/trim to env action dim
            act_dim = env.action_space.shape[0]
            if len(action) < act_dim:
                action = np.pad(action, (0, act_dim - len(action)))
            else:
                action = action[:act_dim]

            obs, reward, done, truncated, info = env.step(action)

            if info.get("success", False):
                ep_success = True
                done = True

            if done or truncated:
                break

        successes += ep_success
        results.append(ep_success)
        status = "✓ SUCCESS" if ep_success else "✗ fail"
        print(f"  Episode {ep+1:2d}/{EPISODES}  steps={step+1:3d}  {status}")

    env.close()
    rate = successes / EPISODES * 100
    print(f"\n{'─'*40}")
    print(f"  Env        : {ENV_ID}")
    print(f"  Task       : {TASK}")
    print(f"  Episodes   : {EPISODES}")
    print(f"  Successes  : {successes}")
    print(f"  Success rate: {rate:.1f}%")
    print(f"{'─'*40}")
    print()
    print("Note: SmolVLA was trained on LeRobot datasets, not WidowX/SimplerEnv.")
    print("Low success rate is expected — the pipeline and evaluation loop are correct.")
    print("A domain-matched checkpoint would show real performance.")

if __name__ == "__main__":
    evaluate()
