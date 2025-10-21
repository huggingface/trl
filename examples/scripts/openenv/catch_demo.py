# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: T201
"""
Simple demo script for the OpenSpiel Catch environment.

This demonstrates the basic workflow:
1. Start the environment server
2. Connect to it
3. Reset and observe initial state
4. Take actions and see rewards
5. Clean up

Setup:

```sh
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
uv pip install open_spiel
```

Usage:

```sh
python trl/experimental/openenv/catch_demo.py
```
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction


ENV_URL = "http://0.0.0.0:8002"


class Policy:
    """Base policy class."""

    def __init__(self, name):
        self.name = name

    def select_action(self, obs):
        """Select an action given an observation."""
        raise NotImplementedError


class RandomPolicy(Policy):
    """Policy that selects random legal actions."""

    def __init__(self):
        super().__init__("🎲 Random Policy")

    def select_action(self, obs):
        import random

        return random.choice(obs.legal_actions) if obs.legal_actions else 0


class LeftPolicy(Policy):
    """Policy that always moves left."""

    def __init__(self):
        super().__init__("⬅️  Always Left")

    def select_action(self, obs):
        return obs.legal_actions[0] if obs.legal_actions else 0


class RightPolicy(Policy):
    """Policy that always moves right."""

    def __init__(self):
        super().__init__("➡️  Always Right")

    def select_action(self, obs):
        return obs.legal_actions[-1] if obs.legal_actions else 2


class SmartPolicy(Policy):
    """Policy that tries to move towards the ball."""

    def __init__(self):
        super().__init__("🧠 Smart Policy")

    def select_action(self, obs):
        # In catch, the info_state contains information about paddle and ball positions
        # This is a simple heuristic - in practice you'd need to understand the state representation
        if not obs.legal_actions:
            return 0

        # For catch game, often legal actions are [0, 1, 2] = [left, stay, right]
        # Simple heuristic: choose middle action (stay) or random
        import random

        return random.choice(obs.legal_actions)


def run_episode(env, policy, visualize=True, delay=0.3):
    """Run one episode with a policy against OpenSpiel environment."""

    # RESET
    result = env.reset()
    obs = result.observation

    if visualize:
        print(f"\n{'=' * 60}")
        print(f"   🎮 {policy.name}")
        print("   🎲 Playing against OpenSpiel Catch")
        print("=" * 60 + "\n")
        time.sleep(delay)

    total_reward = 0
    step = 0
    action_names = ["⬅️  LEFT", "🛑 STAY", "➡️  RIGHT"]

    # THE RL LOOP
    while not obs.done:
        # 1. Policy chooses action
        action_id = policy.select_action(obs)

        # 2. Environment executes (via HTTP!)
        action = OpenSpielAction(action_id=action_id, game_name="catch")
        result = env.step(action)
        obs = result.observation

        # 3. Collect reward
        if result.reward is not None:
            total_reward += result.reward

        if visualize:
            action_name = action_names[action_id] if action_id < len(action_names) else f"ACTION {action_id}"
            print(f"📍 Step {step + 1}: {action_name} → Reward: {result.reward}")
            time.sleep(delay)

        step += 1

    if visualize:
        result_text = "🎉 CAUGHT!" if total_reward > 0 else "😢 MISSED"
        print(f"\n{'=' * 60}")
        print(f"   {result_text} Total Reward: {total_reward}")
        print("=" * 60)

    return total_reward > 0


def start_server():
    """Start the OpenSpiel environment server."""
    print("⚡ Starting FastAPI server for OpenSpiel Catch Environment...")

    work_dir = str(Path.cwd().parent.absolute())

    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8002"],
        env={**os.environ, "PYTHONPATH": f"{work_dir}/src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=work_dir,
    )

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)

    # Check if server is running
    try:
        response = requests.get(f"{ENV_URL}/health", timeout=2)
        print("✅ OpenSpiel Catch Environment server is running!\n")
        return server_process
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        print("\n📋 Checking error output...")
        server_process.poll()
        if server_process.stderr:
            stderr = server_process.stderr.read()
            if stderr:
                print(stderr)
        raise


def run_demo():
    """Run a simple demo of the Catch environment."""
    print("🎯 OpenSpiel Catch Environment Demo")
    print("=" * 60)

    # Connect to environment server
    client = OpenSpielEnv(base_url=ENV_URL)

    try:
        # Reset environment and show initial state
        print("\n📍 Resetting environment...")
        result = client.reset()

        print(f"   Initial observation shape: {len(result.observation.info_state)}")
        print(f"   Info state (first 10 values): {result.observation.info_state[:10]}")
        print(f"   Legal actions: {result.observation.legal_actions}")
        print(f"   Game phase: {result.observation.game_phase}")
        print(f"   Done: {result.done}")
        print(f"   Initial reward: {result.reward}")

        # Demo different policies
        print("\n📺 " + "=" * 64 + " 📺")
        print("   Watch Policies Play Against OpenSpiel!")
        print("📺 " + "=" * 64 + " 📺\n")

        policies = [SmartPolicy(), RandomPolicy(), LeftPolicy(), RightPolicy()]

        for policy in policies:
            caught = run_episode(client, policy, visualize=True, delay=0.5)

        print("\n💡 You just watched REAL OpenSpiel Catch being played!")
        print("   • Every action was an HTTP call")
        print("   • Game logic runs in the server")
        print("   • Client only sends actions and receives observations\n")

        # Get final environment state
        state = client.state()
        print("\n📊 Final Environment State:")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Game: {state.game_name}")
        print(f"   Num players: {state.num_players}")
        print(f"   Agent player: {state.agent_player}")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Always close the environment
        client.close()
        print("\n✅ Demo complete!")


def main():
    """Main function to run the demo."""
    server_process = None
    try:
        server_process = start_server()
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        if server_process:
            print("\n🛑 Shutting down server...")
            server_process.terminate()
            server_process.wait(timeout=5)
        print("👋 Done!")


if __name__ == "__main__":
    main()
