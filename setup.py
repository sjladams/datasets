import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datasets",
    version="0.0.1",
    author="Steven Adams",
    author_email="stevenjladams@gmail.com",
    description="Classification and Regression Pytorch Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjladams/datasets",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9.10",
    install_requires=[
        'torch>=2.5',
        'torchvision>=0.20',
        'uci_datasets @ git+https://github.com/treforevans/uci_datasets.git'
    ],
    package_data={
        'datasets': ['data/**'],
    },
    include_package_data = True
)