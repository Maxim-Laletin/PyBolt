from setuptools import setup, find_packages

setup(
    name="PyBolt",  # Name of your package
    version="0.0.1",  # Initial version of your package
    author="Maxim Laletin, Michał Łukawski, Adam Gomułka",  # Package author
    description="Python framework for the computation of dark matter relic density and energy distribution",
    url="https://github.com/Maxim-Laletin/PyBolt",  # URL to the GitHub repo
    packages=find_packages(),  # Automatically find and include packages in the directory
    install_requires=[  # List of dependencies for your package
        "numpy",
        "matplotlib",
        "scipy",
        "tqdm",
    ],
    classifiers=[  # Additional metadata about your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Python version requirement
)
