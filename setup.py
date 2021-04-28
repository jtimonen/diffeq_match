import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

setuptools.setup(
    name="dem",
    version="0.0.1",
    author="Juho Timonen",
    author_email="juho.timonen@iki.fi",
    description="Generative flows",
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtimonen/diffeq_match",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=["pip>=19.0.3"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
