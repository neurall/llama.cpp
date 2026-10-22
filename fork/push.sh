#!/bin/bash
# push branches to the fork (neurall, plain push, no force) and replace runs.db there with the latest (only commit of branch runs-db, see dbcommit.sh)
# usage: fork/push.sh [branch ...]     no branch = the current one. Never origin, never release.
set -e
R=$(git rev-parse --show-toplevel)
for b in "${@:-$(git -C "$R" branch --show-current)}"; do git -C "$R" push neurall "refs/heads/$b:refs/heads/$b" 2>&1 | tail -1; done
"$R/fork/dbcommit.sh" --push
