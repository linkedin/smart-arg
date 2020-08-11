"""Argument class <=> Human friendly cli"""

from setuptools import find_namespace_packages, setup

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

# TODO uncomment `doc` usages once set up.
doc = 'https://smarg-arg.readthedocs.io'

setup(
    name='smart-arg',
    version='0.0.9',
    author='',
    author_email='',
    description=__doc__,
    package_dir={'': 'src'},
    long_description=readme,
    long_description_content_type='text/markdown',
    license='BSD-2-CLAUSE',
    python_requires='>=3.6',
    # url=doc,
    download_url='https://pypi.python.org/pypi/smart-arg',
    project_urls={
        # 'Documentation': doc,
        'Source': 'https://github.com/linkedin/smart-arg.git',
        'Tracker': 'https://github.com/linkedin/smart-arg/issues',
    },
    packages=find_namespace_packages(where='src'),
    include_package_data=True,
    install_requires=[],
    tests_require=['pytest-flake8'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved',
        "Typing :: Typed"
    ],
    keywords=[
        'typing',
        'argument parser',
        'reverse argument parser',
        'human friendly',
        'configuration (de)serialization',
        'python',
        'cli'
    ]
)
