import setuptools.command.build_py
from setuptools import find_packages, setup

# Set version_info[__version__], while avoiding importing numpy, in case numpy
# and vg are being installed concurrently.
# https://packaging.python.org/guides/single-sourcing-package-version/
version_info = {}
exec(open("lacecore/package_version.py").read(), version_info)

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

with open("requirements_obj.txt") as f:
    obj_requires = f.read()

exclude = ["**/test_*.py"]


class build_py(setuptools.command.build_py.build_py):
    def find_package_modules(self, package, package_dir):
        import fnmatch

        return [
            (pkg, mod, file)
            for (pkg, mod, file) in super().find_package_modules(package, package_dir)
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in exclude)
        ]


setup(
    name="lacecore",
    version=version_info["__version__"],
    description="Polygonal mesh library optimized for cloud computation.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Metabolize, Minnow Software, Body Labs, and other contributors",
    author_email="github@paulmelnikow.com",
    url="https://github.com/lace/lacecore",
    project_urls={
        "Issue Tracker": "https://github.com/lace/lacecore/issues",
        "Documentation": "https://lacecore.readthedocs.io/en/stable/",
    },
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"obj": obj_requires},
    cmdclass={"build_py": build_py},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
