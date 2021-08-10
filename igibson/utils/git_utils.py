from pathlib import Path

# import bddl
import git

import igibson


def git_info(directory):
    repo = git.Repo(directory)
    try:
        branch_name = repo.active_branch.name
    except TypeError:
        branch_name = "[DETACHED]"
    return {
        "directory": str(directory),
        "code_diff": repo.git.diff(None),
        "code_diff_staged": repo.git.diff("--staged"),
        "commit_hash": repo.head.commit.hexsha,
        "branch_name": branch_name,
    }


def project_git_info():
    return {
        "iGibson": git_info(Path(igibson.root_path).parent),
        # TODO: add version to bddl, assets, ig_dataset
        # "bddl": git_info(Path(bddl.__file__).parent.parent),
        # "ig_assets": git_info(igibson.assets_path),
        # "ig_dataset": git_info(igibson.ig_dataset_path),
    }
