import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "numpy",
    "scipy",
    "tqdm",
]

test_requirements = [
    "pytest",
]

setuptools.setup(
    name="rdot",
    version="0.0.2",
    author="Nathaniel Imel",
    author_email="nimel@uci.edu",
    description="Rate distortion optimization tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathimel/rdot",
    project_urls={"Bug Tracker": "https://github.com/nathimel/rdot/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=requirements,
    extra_requires={"test": test_requirements},
    #    python_requires=">=3.10.6",  # Colab-compatible
)
