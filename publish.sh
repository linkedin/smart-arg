star_version=$1
version_file=$2
sdist_package_name=$3

VERSION=$(bash ./resolve_version.sh "$star_version")
sed -ie "s/^__version__ *=.*$/__version__ = '$VERSION'/g" "$version_file"

rm -rf dist
python setup.py sdist
twine upload dist/* --verbose

# VERSION may be normalized by sdist
VERSION=$(ls dist|sed -E "s/^$sdist_package_name-(.*)\.tar\.gz$/v\1/")
bash tag.sh "$VERSION"