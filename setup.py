from setuptools import setup, find_packages

setup(
    name="rl_coppelia",
    version="0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # src is the root of the package
    install_requires=[
        "argparse", "stable_baselines3", "tensorboard"
    ],
    entry_points={
        "console_scripts": [
            "rl_coppelia=rl_coppelia.cli:main"
        ]
    },
)