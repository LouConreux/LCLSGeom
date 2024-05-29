from setuptools import setup, find_packages

setup(
    name="LCLSGeom",
    version="0.1.0",
    author="Louis Conreyx",
    author_email="louis.conreux1@gmail.com",
    description="Miscellaneous functions for converting geometry files for LCLS geometry calibrations and optimizations.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/LouConreux/LCLSGeom",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    license="SLAC Internal Use Only",
    license_files=('LICENSE',),
    python_requires='>=3.6',
)
