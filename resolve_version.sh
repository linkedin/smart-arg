VERSION=$1

# detect if local
[ "x$GITHUB_TOKEN" == "x" ] && local_version=.dev0

if [[ "$VERSION" =~ ^([0-9]+\.){2,3}\*$ ]]; then
  star_version_prefix=${VERSION%%\*}

  last_matched_tag=$(git describe --match "v$VERSION" --tags HEAD --first-parent --abbrev=0 2>/dev/null)
  [ $? -eq 0 ] && patch_version=$((${last_matched_tag##*[!0-9]} + 1));

  resolved_version=$star_version_prefix$((patch_version))$local_version
elif [[ "$VERSION" =~ \* ]]; then
  printf "Unsupported star version: '%s'\nOnly supports star patch version or subpatch/4th version." "$VERSION" >&2
  exit 128
else
  resolved_version=$VERSION
fi

echo "$resolved_version"
echo "Version is resolved to '$resolved_version'." >&2