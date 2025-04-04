# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""trl is an open library for RL with transformer models.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for PyPI.

0. Prerequisites:
   - Dependencies:
     - twine: "pip install twine"
   - Create an account in (and join the 'trl' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/

1. Change the version in:
   - __init__.py
   - setup.py

2. Commit these changes: "git commit -m 'Release: VERSION'"

3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for PyPI'"
   Push the tag to remote: git push --tags origin main

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   First, delete any "build" directory that may exist from previous builds.

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the PyPI test server:

   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv/notebook by running:
   pip install -i https://testpypi.python.org/pypi trl

6. Upload the final version to actual PyPI:
   twine upload dist/* -r pypi

7. Fill release notes in the tag in github once everything is looking hunky-dory.

8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
   Then push the change with a message 'set dev version'
"""

from setuptools import find_packages, setup


__version__ = "0.16.1"  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)

REQUIRED_PKGS = [
    "accelerate>=0.34.0",
    "datasets>=3.0.0",
    "rich",  # rich shouldn't be a required package for trl, we should remove it from here
    "transformers>=4.46.0",
]
EXTRAS = {
    # Windows support is partially supported with DeepSpeed https://github.com/microsoft/DeepSpeed/tree/master#windows
    "deepspeed": ["deepspeed>=0.14.4; sys_platform != 'win32'"],
    "diffusers": ["diffusers>=0.18.0"],
    "judges": ["openai>=1.23.2", "llm-blender>=0.0.2"],
    # liger-kernel depends on triton, which is only available on Linux https://github.com/triton-lang/triton#compatibility
    # can be set to >=0.5.3 when https://github.com/linkedin/Liger-Kernel/issues/586 is fixed
    "liger": ["liger-kernel==0.5.3; sys_platform != 'win32'"],
    "mergekit": ["mergekit>=0.0.5.1"],
    "peft": ["peft>=0.8.0"],
    "quantization": ["bitsandbytes"],
    "scikit": ["scikit-learn"],
    "test": ["parameterized", "pytest-cov", "pytest-rerunfailures", "pytest-xdist", "pytest"],
    # vllm is not available on Windows
    "vllm": ["vllm>=0.7.0; sys_platform != 'win32'", "fastapi", "pydantic", "requests", "uvicorn"],
    "vlm": ["Pillow"],
}
EXTRAS["dev"] = []
for reqs in EXTRAS.values():
    EXTRAS["dev"].extend(reqs)


setup(
    name="trl",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/huggingface/trl",
    entry_points={
        "console_scripts": ["trl=trl.cli:main"],
    },
    include_package_data=True,
    package_data={
        "trl": ["templates/*.md"],
    },
    packages=find_packages(exclude={"tests", "tests.slow", "trl.templates"}),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    python_requires=">=3.9",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    version=__version__,
    description="Train transformer language models with reinforcement learning.",
    keywords="transformers, huggingface, language modeling, post-training, rlhf, sft, dpo, grpo",
    author="Leandro von Werra",
    author_email="leandro.vonwerra@gmail.com",
)
