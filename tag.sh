# get repo name from git
remote=$(git config --get remote.origin.url)
repo=$(basename "$remote" .git)

commit=$(git rev-parse HEAD)
new=$1

# Simulate the tagging locally and populate the error
git tag "$new" || exit $?

# POST a new ref to repo via Github API
curl -s -X POST "https://api.github.com/repos/$REPO_OWNER/$repo/git/refs" \
-H "Authorization: token $GITHUB_TOKEN" \
-d @- << EOF
{
  "ref": "refs/tags/$new",
  "sha": "$commit"
}
EOF