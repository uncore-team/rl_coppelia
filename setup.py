from setuptools import setup, find_packages

setup(
    name="rl_coppelia",
    version="0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # src is the root of the package
    install_requires=[
        "argparse", "stable_baselines3", "tensorboard", "PyQt5"
    ],
    entry_points={
        "console_scripts": [
            "rl_coppelia=rl_coppelia.cli:main",
            "rl_coppelia_gui=rl_coppelia.gui_rl_coppelia:main"
        ]
    },
    include_package_data=True,
    package_data={
        "rl_coppelia": ["assets/*.png"],
    },

)