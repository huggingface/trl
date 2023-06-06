import subprocess


def test_dqn_jax():
    subprocess.run(
        "python examples/hello_world.py",
        shell=True,
        check=True,
    )
