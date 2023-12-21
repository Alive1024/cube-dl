import warnings

import git
from git import InvalidGitRepositoryError


def auto_commit(cwd: str, dst_branch_name: str = "cube"):
    # Check whether `cwd` is a Git repo
    try:
        git.Repo(cwd)
    except InvalidGitRepositoryError:
        warnings.warn(f'"{cwd}" is not a Git repo. Please initialize it manually if you hope auto-commit to work.')
        return

    repo = git.Repo(cwd)
    # if not repo.is_dirty():
    #     return

    active_branch_name = repo.active_branch.name

    branch_names = [branch.name for branch in repo.branches]
    # Create a new branch with name `branch_name` if it doesn't exist
    if dst_branch_name not in branch_names:
        # repo.git.checkout("-b", branch_name)
        repo.git.branch(dst_branch_name)

    if repo.is_dirty():
        print(repo.git.stash())  # `git stash`: stash current changes
        repo.git.checkout(dst_branch_name)
        repo.git.merge(active_branch_name)  #
        repo.git.add(".")
        repo.git.commit(m="Auto commit by cube")
        # print(ret)
        repo.git.checkout(active_branch_name)
    else:
        print("clean")
        repo.git.checkout(dst_branch_name)
        repo.git.merge(active_branch_name)  # merge active branch into the dst branch
        repo.git.checkout(active_branch_name)

    return dst_branch_name
