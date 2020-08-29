"""Argument class <=> Human friendly cli"""

from setuptools import setup
from smart_arg import __version__


def _resolve_version():
    import subprocess
    is_dynamic = __version__[-1] == '*'
    if is_dynamic:
        kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'shell': True}
        completed_process = subprocess.run(f'git describe --tags --match "v{__version__}"', **kwargs)
        git_tag = completed_process.stdout.decode('utf-8').rstrip('\n')
        patch = git_tag[git_tag.rfind('-', 0, -9) + 1:-9] if git_tag else '0'
        version = __version__[0:-1] + patch
    else:
        version = __version__
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
    tests_require=['pytest'],
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
