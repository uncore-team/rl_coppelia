from setuptools import setup, find_packages

setup(
    name="rl_coppelia",
    version="0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # src is the root of the package
    # install_requires=[
    #     "argcomplete", "stable_baselines3", "tensorboard", "PyQt5"
    # ],
    entry_points={
        "console_scripts": [
            "uncore_rl=rl_coppelia.cli:main",
            "uncore_rl_gui=gui.main:main"
        ]
    },
    include_package_data=True,
    package_data={
        "rl_coppelia": ["assets/*.png"],
    },

)