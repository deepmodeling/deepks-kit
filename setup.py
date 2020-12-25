import pathlib
import setuptools


here = pathlib.Path(__file__).parent.resolve()
readme = (here / 'README.md').read_text(encoding='utf-8')

# did not include torch and pyscf here
install_requires=['numpy', 'paramiko', 'ruamel.yaml']


setuptools.setup(
    name="deepks",
    use_scm_version={'write_to': 'deepks/_version.py'},
    setup_requires=['setuptools_scm'],
    author="Yixiao Chen",
    author_email="yixiaoc@princeton.edu",
    description="DeePKS-kit: generate accurate (self-consistent) energy functionals",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['deepks', 'deepks.*']),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    keywords='deepks DeePKS-kit',
    install_requires=install_requires,
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'deepks=deepks.main:main_cli',
        ],
    },
)