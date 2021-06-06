import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_lightning_template",
    version="0.1.0",
    author="Sam Purkis",
    author_email="sam.purkis@hotmail.co.uk",
    description="A thin wrapper for Pytorch Lightning, inspired by Keras API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smpurkis/autocompile",
    project_urls={
        "Bug Tracker": "https://github.com/smpurkis/autocompile/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'pytorch-lightning',
        'torch',
        'torchinfo'
    ],
)