set -e

function git_tag() {
  # get repo name from git
  repo=$(git config --get remote.origin.url | sed 's|^.*[/:]\(.*/.*\)\(\.git\)\?$|\1|')
  commit=$(git rev-parse HEAD)
  new_tag=$1
  git tag "$new_tag"

  echo >&2 POST the new tag ref "$new_tag" to "$repo" via Github API
  HTTP_STATUS=$(
    curl -w "%{http_code}" -o >(cat >&2) -si -X POST "https://api.github.com/repos/$repo/git/refs" \
      -H "Authorization: token $GITHUB_TOKEN" \
      -d @- <<EOF
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
  if [[ "$star_version" =~ ^(([0-9]+\.){1,2})(\*|[0-9]+)$ ]]; then
    if [[ ${BASH_REMATCH[3]} == "*" ]]; then
      star_version_prefix=${BASH_REMATCH[1]}

      # expect all tags are reachable from HEAD and only one version tag per commit
      if last_matched_tag=$(git describe --match "v$star_version" --tags HEAD --first-parent --abbrev=0 2>/dev/null); then
        echo >&2 last_matched_tag is "$last_matched_tag" for "$star_version"
        rest=${last_matched_tag#v$star_version}
        increment_version=$((${rest%%.*} + 1))
        [ "$(git rev-parse HEAD)" == "$(git rev-parse "$last_matched_tag")" ] && echo HEAD already at "$last_matched_tag" >&2 && return 126
      fi
      resolved_version=$star_version_prefix$((increment_version))
      echo "$star_version is resolved to version '$resolved_version'." >&2
      echo "$resolved_version"
    else
      echo "Using the input star version '$star_version' as a fixed version." >&2
      echo "$star_version"
    fi
  else
    echo -e "Unsupported star version: '$star_version'.\nOnly supports star minor or patch version." >&2
    return 127
  fi
}

# Set star_version to the first argument or $STAR_VERSION or $($STAR_VERSION_CMD) in this order
star_version=${1-${STAR_VERSION-$($STAR_VERSION_CMD)}}
new_version=$(resolve_version "$star_version")
git_tag v"$new_version"

rm -rf dist
python setup.py sdist
twine upload dist/* --verbose
