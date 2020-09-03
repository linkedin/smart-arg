"""Argument class <=> Human friendly cli"""

from setuptools import setup
from smart_arg import __version__


def _resolve_version():
    import subprocess
    if __version__[-1] == '*':  # Only take the last char as '*' as convention, not checking for errors here
        base_tag = 'v' + __version__[0:-1] + '0'
        run = subprocess.run(f'git rev-list --first-parent --count "{base_tag}"..', stdout=subprocess.PIPE, shell=True)
        version = __version__[0:-1] + ('0' if run.returncode else run.stdout.decode('utf-8').rstrip('\n'))
    else:
        version = __version__
    print(f"Version is resolved to {version!r}.")
    return version


with open('README.md', encoding='utf-8') as f:
    readme = f.read()

# TODO uncomment `doc` usages once set up.
doc = 'https://smart-arg.readthedocs.io'

setup(
    name='smart-arg',
    version=_resolve_version(),
    description=__doc__,
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
    py_modules=['smart_arg'],
    install_requires=[],
    tests_require=['pytest', 'mypy'],
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
