from setuptools import setup, find_packages

setup(
    name="lofarnn",
    version="0.7",
    license="GPLv3",
    author="Jacob Bieker",
    authoer_email="jacob@bieker.tech",
    url="https://github.com/jacobbieker/lofarnn",
    download_url="https://github.com/jacobbieker/lofarnn/archive/v0.6.0.tar.gz",
    keywords=["Radio Astronomy", "PyTorch", "Machine Learning"],
    packages=find_packages(),
    install_requires=["astropy", "numpy", "scikit-image", "pillow"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
