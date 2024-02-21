# Copyright (c) KEG, Tsinghua University.
#
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import find_packages, setup


def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="MMDoctor",
    version='0.2.0',
    description="",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={},
    packages=find_packages(),
    url="https://dev.aminer.cn/yuwenmeng/mm-doctor",
    author="THUKEG",
    author_email="",
    scripts={},
    include_package_data=True,
    python_requires=">=3.5",
    license="Apache 2.0 license"
)