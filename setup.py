from setuptools import setup, find_packages

setup(
    name="modules_FNA",
    version="0.1",
    packages=find_packages(include=["FlexCNN_for_Medical_Physics", "FlexCNN_for_Medical_Physics.*"]),
    install_requires=[
        # Add external dependencies here, e.g. "numpy", "torch"
    ],
)