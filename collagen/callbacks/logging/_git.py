import os
import subprocess

__all__ = ["git_info"]


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]


# Return the git revision as a string
def git_info():
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')

        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_branch = out.strip().decode('ascii')
    except OSError:
        return None

    return git_branch, git_revision
