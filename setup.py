"""Argument class <=> Human friendly cli"""

from setuptools import setup

doc = 'https://smart-arg.readthedocs.io'
with open('README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='smart-arg',
    use_scm_version={
        'root': '.',
        'relative_to': __file__
    },
    setup_requires=['setuptools_scm'],
    description=__doc__,
    long_description=readme,
    long_description_content_type='text/markdown',
    license='BSD-2-CLAUSE',
    python_requires='>=3.6',
    url=doc,
    download_url='https://pypi.python.org/pypi/smart-arg',
    project_urls={
        'Documentation': doc,
        'Source': 'https://github.com/linkedin/smart-arg.git',
        'Tracker': 'https://github.com/linkedin/smart-arg/issues',
    },
    py_modules=['smart_arg'],
    install_requires=[],
    tests_require=['pytest', 'mypy'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved',
        'Typing :: Typed'
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
