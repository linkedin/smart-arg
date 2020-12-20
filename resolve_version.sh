star_version=$1

# detect if local
[ "x$GITHUB_TOKEN" == "x" ] && local_version=.dev0

if [[ "$star_version" =~ ^([0-9]+\.){2}\*$ ]]; then
  star_version_prefix=${star_version%%\*}

  # expect all tags are reachable from HEAD and only one version tag per commit
  last_matched_tag=$(git describe --match "v$star_version" --tags HEAD --first-parent --abbrev=0 2>/dev/null)
  [ $? -eq 0 ] && patch_version=$((${last_matched_tag##*[!0-9]} + 1));

  resolved_version=$star_version_prefix$((patch_version))$local_version
elif [[ "$star_version" =~ \* ]]; then
  printf "Unsupported star version: '%s'\nOnly supports star patch version." "$star_version" >&2
  exit 128
else
  resolved_version=$star_version
fi

echo "$resolved_version"
echo "Version is resolved to '$resolved_version'." >&2