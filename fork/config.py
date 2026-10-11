"""Machine-specific settings of the fork tools (paths, model files, W10 build host), kept out of the code.

Defaults suit a plain Linux box (everything under ~/llama-fork, models under ~/models). Override in a TOML file, first found wins:
  $FORK_CONFIG, ./fork/local.toml (ignored by git), ~/.config/llama-fork/config.toml
and per setting with the environment variable FORK_<SECTION>_<KEY> (upper case), e.g. FORK_PATHS_RELS=/data/rels.

  [paths]   rels  gh_rel  devbuilds  release_out  images  best  ccache  ccache_stock  build_repo  build_tmp  work  agent
  [models]  olmoe  qwen36  qwen38  glm  mimo        (the regress cells and baseline.py skip a model whose file is missing)
  [hosts]   w10_ssh  w10_root                         (the Windows laptop used by `run.py build --machine pc3`; the ssh target may be an alias from ~/.ssh/config)
  [gpu]     nvidia_smi  limits_env  limits_service  env_sh   (power-limit tools plcal.py / baseline.py: nvidia-smi to call through sudo, the files that keep the limits at boot, an
                                                              optional shell file whose exports every queue task gets; empty = not used)
"""
import os

HOME = os.path.expanduser("~")
_BASE = os.path.join(HOME, "llama-fork")
DEFAULTS = {
    "paths": {
        "rels": _BASE + "/rels", "gh_rel": _BASE + "/stock", "devbuilds": _BASE + "/rels/devbuilds", "release_out": _BASE + "/rels/release-out",
        "images": _BASE + "/rels/images", "best": _BASE + "/rels/best", "ccache": _BASE + "/ccache", "ccache_stock": _BASE + "/ccache-stock",
        "build_repo": os.getcwd(), "build_tmp": _BASE + "/build-tmp", "work": _BASE + "/work", "agent": "",
    },
    "models": {
        "olmoe": HOME + "/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        "qwen36": HOME + "/models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf",
        "qwen38": HOME + "/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        "glm": HOME + "/models/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
        "mimo": HOME + "/models/MiMo-V2.6-Flash-MOPD-MXFP4.gguf",
    },
    "hosts": {"w10_ssh": "", "w10_root": "N:"},
    "gpu": {"nvidia_smi": "nvidia-smi", "limits_env": "", "limits_service": "", "env_sh": ""},
}


def _load():
    import tomllib
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.environ.get("FORK_CONFIG"), os.path.join(here, "local.toml"), os.path.join(HOME, ".config", "llama-fork", "config.toml")):
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                return tomllib.load(f), p
    return {}, None


_FILE, SOURCE = _load()


def get(section, key):
    """environment FORK_<SECTION>_<KEY>, then the TOML file, then the default"""
    env = os.environ.get(f"FORK_{section}_{key}".upper())
    if env is not None:
        return env
    if key in _FILE.get(section, {}):
        return _FILE[section][key]
    return DEFAULTS[section][key]


def path(key):
    return os.path.expanduser(get("paths", key))


def model(key):
    return os.path.expanduser(get("models", key))
