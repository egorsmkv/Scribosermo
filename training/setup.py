import setuptools

setuptools.setup(
    name="scode",
    version="0.0.1",
    author="Jaco Erithacus",
    author_email="jaco@mail.de",
    description="Library for nicer import statements",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "num2words",
        "pandarallel",
        "pandas",
        "pyyaml",
        "tensorflow<2.5,>=2.4",
        "tensorflow-io<0.18",
        "tensorflow-addons<0.13",
        "tqdm",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={"": ["data/*"]},
)
