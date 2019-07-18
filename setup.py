from setuptools import find_packages, setup


with open("README.md") as f:
    readme = f.read()

extras = {"test": ["pytest>=4.6.3"]}

setup(
    author="Chris Lloyd Barnes",
    author_email="barnesc@janelia.hhmi.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    description="Abstract base classes for making h5py-like objects.",
    install_requires=["numpy"],
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    extras_require=extras,
    keywords="h5py",
    name="h5py_like",
    packages=find_packages(exclude=["tests"]),
    url="https://github.com/barnesc/h5py_like",
    version="0.5.0",
    zip_safe=False,
)
