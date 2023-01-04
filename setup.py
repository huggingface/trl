from setuptools import setup, find_packages

__version__ = "0.1.1"  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)

REQUIRED_PKGS = [
    "torch>=1.4.0",
    "transformers>=4.18.0",
    "numpy>=1.18.2",
]
EXTRAS = {
    "test" : ["pytest","pytest-xdist","accelerate", "datasets", "wandb"],
    "dev" : ["pytest","pytest-xdist", "black", "isort", "flake8>=3.8.3", "accelerate", "datasets", "wandb"],
}

setup(
    name="trl",
    license="Apache 2.0",
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            'Natural Language :: English',
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    url="https://github.com/lvwerra/trl",
    packages = find_packages(),
    include_package_data = True,
    install_requires = REQUIRED_PKGS,
    extras_require = EXTRAS,
    python_requires = '>=3.6',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    zip_safe = False,
    version=__version__,
    description="A Pytorch implementation of Proximal Policy Optimization for transfomer language models.",
    keywords="ppo, transformers, huggingface, gpt2, language modeling, rlhf",
    author="Leandro von Werra",
    author_email="leandro.vonwerra@gmail.com",
)
