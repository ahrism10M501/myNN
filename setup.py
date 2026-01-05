from setuptools import setup, find_packages

setup(
    name="myNN",
    version="0.1.0",
    author="ahrism7",
    author_email="ariofeem@gmail.com",
    description="NN library",
    url="https://github.io/ahrism10M501/myNN",
    packages=find_packages(),
    
    install_requires=[
        "numpy>=2.2.6"
    ],
    
    python_requires='>=3.10'
)