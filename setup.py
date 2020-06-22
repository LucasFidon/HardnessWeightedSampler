import os
import setuptools

about = {}
with open(os.path.join("hardness_weighted_sampler", "__about__.py")) as fp:
    exec(fp.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

def install_requires(fname="requirements.txt"):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

setuptools.setup(
    name="Hardness Weighted Sampler",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__summary__"],
    license=about["__license__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasFidon/HardnessWeightedSampler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires = install_requires(),
    keywords='deeplearning hardexamplemining',
)