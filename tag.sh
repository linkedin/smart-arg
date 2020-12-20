# get repo name from git
remote=$(git config --get remote.origin.url)
repo=$(basename "$remote" .git)

commit=$(git rev-parse HEAD)
new_tag=$1

# Tag locally and populate error if any
git tag "$new_tag" || exit $?

# POST the new tag ref to repo via Github API
1>&2 curl -s -X POST "https://api.github.com/repos/$REPO_OWNER/$repo/git/refs" \
-H "Authorization: token $GITHUB_TOKEN" \
-d @- << EOF
{
  "ref": "refs/tags/$new_tag",
  "sha": "$commit"
}
EOF