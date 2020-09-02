import pathlib
import setuptools


here = pathlib.Path(__file__).parent.resolve()
readme = (here / 'README.md').read_text(encoding='utf-8')

# did not include torch and pyscf here
install_requires=['numpy', 'paramiko', 'ruamel.yaml']


setuptools.setup(
    name="deepqc",
    use_scm_version={'write_to': 'deepqc/_version.py'},
    setup_requires=['setuptools_scm'],
    author="Yixiao Chen",
    author_email="yixiaoc@princeton.edu",
    description="DeePQC: generate correlation energy functionals for chemical systems",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['deepqc', 'deepqc.*']),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    keywords='deepqc quantum chemistry',
    install_requires=install_requires,
    python_requires="~=3.7",
    entry_points={
        'console_scripts': [
            'deepqc=deepqc.main:cli',
        ],
    },
)