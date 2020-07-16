from setuptools import find_namespace_packages, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
# TODO add project_urls once determined.
setup(
    name='smart-arg',
    version='1.0.0',
    author='',
    author_email='',
    description='Smart Arguments Suite',
    package_dir={'': 'src'},
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    download_url='',
    packages=find_namespace_packages(where='src'),
    include_package_data=True,
    install_requires=[],
    tests_require=['pytest-flake8'],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        "Typing :: Typed"
    ],
    keywords=[
        'typing',
        'argument parser',
        'python'
    ]
)
