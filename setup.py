import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="sent2vec",
    version="0.0.1",
    description="How to represent sentences in a high-dimensional vector space, a.k.a., sentence embedding.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pdrm83/sent2vect",
    author="Pedram Ataee",
    author_email="pedram.ataee@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["sent2vec"],
    include_package_data=True,
    install_requires=['transformers', 'torch', 'numpy', 'pandas'],
    entry_points={
        "console_scripts": [
            "pdrm83=sent2vec.__main__:main",
        ]
    },
)