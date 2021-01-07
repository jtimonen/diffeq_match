import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dem",
    version="0.0.1",
    author="Juho Timonen",
    author_email="juho.timonen@iki.fi",
    description="Generative flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtimonen/diffeq_match",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

