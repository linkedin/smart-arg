set -e
function tag_remote() {
  # get repo name from git
  remote=$(git config --get remote.origin.url | sed "s|:|/|g")
  project=$(basename "$remote" .git)
  repo=$(basename $(dirname "$remote"))/$project

  commit=$(git rev-parse HEAD)
  new_tag=$1

  # POST the new tag ref to repo via Github API
  HTTP_STATUS=$(curl -w "%{http_code}" -o >(cat >&2) -si -X POST "https://api.github.com/repos/$repo/git/refs" \
  -H "Authorization: token $GITHUB_TOKEN" \
  -d @- << EOF
  {
    "ref": "refs/tags/$new_tag",
    "sha": "$commit"
  }
EOF
)
  [ "$HTTP_STATUS" == "201" ]
}

function resolve_version() {
  star_version=$1

  if [[ "$star_version" =~ ^([0-9]+\.){2}\*$ ]]; then
    star_version_prefix=${star_version%%\*}

    # expect all tags are reachable from HEAD and only one version tag per commit
    last_matched_tag=$(git describe --match "v$star_version" --tags HEAD --first-parent --abbrev=0 2>/dev/null)
    [ $? -ne 0 ] || patch_version=$((${last_matched_tag##*[!0-9]} + 1));

    resolved_version=$star_version_prefix$((patch_version))
  elif [[ "$star_version" =~ ^([0-9]+\.){2}[0-9]+$ ]]; then
    resolved_version=$star_version
  else
    printf "Unsupported star version: '%s'\nOnly supports star patch version." "$star_version" >&2
    return 127
  fi

  echo "$star_version is resolved to version '$resolved_version'." >&2
  echo "$resolved_version"
}

# Set star_version to the first argument or $STAR_VERSION or $($STAR_VERSION_CMD) in this order
star_version=${1-${STAR_VERSION-$($STAR_VERSION_CMD)}}
version=v$(resolve_version "$star_version")
git tag "$version"  # for setuptools-scm

# version may be normalized by setuptools
normalized_version=v$(python setup.py --version)
[ "$version" == "$normalized_version" ] || ( git tag "$normalized_version" && git tag -d "$version" )
[ "$GITHUB_TOKEN" != "" ] && tag_remote "$normalized_version"

rm -rf dist
python setup.py sdist
twine upload dist/* --verbose