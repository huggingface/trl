import subprocess


def test_hello_world():
    subprocess.run(
        "python examples/hello_world.py",
        shell=True,
        check=True,
    )
