import time
from echo_env import EchoEnv
from echo_env.models import EchoAction

class MyEchoEnv:
    def __init__(self):
        self.env = EchoEnv(base_url="https://openenv-echo-env.hf.space")

    def step(self, message: str) -> str:
        """
        Echo the message back from the environment.

        Args:
            message: The message to echo

        Returns:
            The echoed message.
        """
        observation = self.env.step(EchoAction(message=message))
        return observation.observation.echoed_message

def main():
    print(f"🌍 Using Hugging Face Space environment at: 0.0.0.0")
    env = MyEchoEnv()
    for i in range(10):
        env_result = env.step(f"Hello {i}")
        print(f"Step {i} - Env Result: {env_result}")
    time.sleep(5)


if __name__ == "__main__":
    main()
