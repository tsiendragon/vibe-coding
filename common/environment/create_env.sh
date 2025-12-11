#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
ENV_NAME="${ENV_NAME:-clmf}"
PY_VER="${PY_VER:-3.12.*}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/miniforge3}"
YML_PATH="${YML_PATH:-environment.yml}"

# ---------- Utils ----------
log()  { printf "\033[1;34m[*]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[OK]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR]\033[0m %s\n" "$*"; }
need_cmd(){ command -v "$1" >/dev/null 2>&1 || { err "Missing '$1'"; exit 1; }; }
is_macos(){ [[ "$(uname -s)" == "Darwin" ]]; }

detect_installer(){
  local OS="$(uname -s)" ARCH="$(uname -m)"
  case "$OS" in
    Darwin) [[ "$ARCH" = "arm64" ]] && echo "Miniforge3-MacOSX-arm64.sh" || echo "Miniforge3-MacOSX-x86_64.sh" ;;
    Linux)  [[ "$ARCH" = "aarch64" ]] && echo "Miniforge3-Linux-aarch64.sh" || echo "Miniforge3-Linux-x86_64.sh" ;;
    *) err "Unsupported OS $OS" ;;
  esac
}

fetch(){
  if command -v curl >/dev/null 2>&1; then curl -L "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -O "$2" "$1"
  else err "Need curl or wget"; fi
}

# ---------- Miniforge / mamba ----------
ensure_miniforge(){
  if [[ -x "${INSTALL_DIR}/bin/conda" ]]; then ok "Miniforge already at ${INSTALL_DIR}"; return; fi
  local F URL TMP="/tmp/miniforge.sh"
  F="$(detect_installer)"
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/${F}"
  log "Downloading ${URL}"
  fetch "${URL}" "${TMP}"
  bash "${TMP}" -b -p "${INSTALL_DIR}"
  rm -f "${TMP}"
  ok "Miniforge installed"
}

init_shell(){
  # shellcheck disable=SC1091
  eval "$("${INSTALL_DIR}/bin/conda" shell.bash hook)"
  conda config --set channel_priority strict
  conda config --add channels conda-forge || true
  ok "Conda initialized"
}

ensure_mamba(){
  if "${INSTALL_DIR}/bin/conda" list -n base | grep -q "^mamba\s"; then ok "mamba present"; else
    log "Installing mamba into base"
    conda install -y -n base -c conda-forge mamba
    ok "mamba installed"
  fi
}

# ---------- YAML parsing ----------
# 重建只包含 conda 依赖的 yml：name/channels/dependencies（不含 pip 块）
build_conda_only_yml(){
  local SRC="${YML_PATH}"
  local OUT="/tmp/conda_only_${ENV_NAME}.yml"
  [[ -f "${SRC}" ]] || { err "Missing ${SRC}"; }

  # 取 channels（若未写则默认 conda-forge）
  local channels=()
  awk '
    BEGIN{in_ch=0}
    /^[[:space:]]*channels[[:space:]]*:/ {in_ch=1; next}
    in_ch==1 {
      if ($0 ~ /^[[:space:]]*-/) {
        line=$0; sub(/^[[:space:]]*-[[:space:]]*/,"",line); sub(/[[:space:]]*#.*/,"",line);
        gsub(/^[[:space:]]+|[[:space:]]+$/,"",line); if(length(line)>0) print line;
      } else { exit }
    }
  ' "${SRC}" | while read -r c; do channels+=("$c"); done
  if (( ${#channels[@]} == 0 )); then channels=("conda-forge"); fi

  # 取 dependencies（只保留非 pip 行）
  local deps=()
  awk '
    function lsp(s){ gsub(/[^ ].*$/,"",s); return length(s) }
    BEGIN{in_dep=0; in_pip=0; pip_indent=-1}
    /^[[:space:]]*dependencies[[:space:]]*:/ {in_dep=1; next}
    {
      if(in_dep==1){
        if ($0 ~ /^[[:space:]]*-[[:space:]]*pip[[:space:]]*:/){ in_pip=1; pip_indent=lsp($0); next }
        if (in_pip==1){ if (lsp($0) > pip_indent) next; else in_pip=0 }
        if ($0 ~ /^[[:space:]]*-[[:space:]]*[^#]+/){
          line=$0; sub(/^[[:space:]]*-[[:space:]]*/,"",line); sub(/[[:space:]]*#.*/,"",line);
          gsub(/^[[:space:]]+|[[:space:]]+$/,"",line);
          if(length(line)>0) print line;
        } else if ($0 ~ /^[[:alnum:]_]+:/){ exit }
      }
    }
  ' "${SRC}" | while read -r d; do deps+=("$d"); done

  # 写出规范 yml（固定缩进）
  {
    printf "name: %s\n" "${ENV_NAME}"
    printf "channels:\n"; for c in "${channels[@]}"; do printf "  - %s\n" "$c"; done
    printf "dependencies:\n"
    printf "  - python=%s\n" "${PY_VER}"
    printf "  - pip\n  - setuptools\n  - wheel\n"
    for d in "${deps[@]}"; do printf "  - %s\n" "$d"; done
  } > "${OUT}"

  echo "${OUT}"
}

# 提取 pip 列表（保持顺序，带参数，如 --index-url）
extract_pip_list(){
  [[ -f "${YML_PATH}" ]] || return 0
  awk '
    function lsp(s){ gsub(/[^ ].*$/,"",s); return length(s) }
    BEGIN{in_pip=0; pip_indent=-1}
    {
      if ($0 ~ /^[[:space:]]*-[[:space:]]*pip[[:space:]]*:/){ in_pip=1; pip_indent=lsp($0); next }
      if (in_pip==1){
        if (lsp($0) > pip_indent){
          if ($0 ~ /^[[:space:]]*-[[:space:]]*[^#]+/){
            line=$0
            sub(/^[[:space:]]*-[[:space:]]*/,"",line)   # 去掉前导 "- "
            sub(/[[:space:]]*#.*/,"",line)               # 去掉行尾注释
            gsub(/^[[:space:]]+|[[:space:]]+$/,"",line)  # trim
            if(length(line)>0) print line
          }
          next
        } else { in_pip=0 }
      }
    }
  ' "${YML_PATH}"
}

# ---------- Env ops ----------
create_env_if_needed(){
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then ok "Env ${ENV_NAME} exists"; else
    log "Creating env '${ENV_NAME}' with python=${PY_VER}"
    mamba create -y -n "${ENV_NAME}" "python=${PY_VER}"
    ok "Env created"
  fi
}

apply_conda_only(){
  local F="$1"
  log "Applying conda-only ${F} → ${ENV_NAME}"
  mamba env update -n "${ENV_NAME}" -f "${F}" --prune
  ok "Conda deps applied"
}

install_pip_seq(){
  local pkgs=()
  while IFS= read -r p; do [[ -z "$p" ]] && continue; pkgs+=("$p"); done < <(extract_pip_list)
  (( ${#pkgs[@]} == 0 )) && { warn "No pip packages in ${YML_PATH}"; return; }

  log "Installing pip packages sequentially (${#pkgs[@]})"
  for p in "${pkgs[@]}"; do
    # macOS: 把任何 cu121/cu128 索引改为 cpu
    if is_macos; then
      p="$(echo "$p" | sed -E 's|https://download\.pytorch\.org/whl/cu(121|128)|https://download.pytorch.org/whl/cpu|g')"
    fi
    log "pip install -U ${p}"
    # 用 bash -lc 让带参数的行（如 torch --index-url ...）保持正确拆分
    conda run -n "${ENV_NAME}" bash -lc "python -m pip install -U ${p}"
  done
  ok "Pip packages installed"
}

pin_python_again(){
  log "Re-affirm python=${PY_VER}"
  mamba install -y -n "${ENV_NAME}" "python=${PY_VER}" --freeze-installed
  ok "Python pinned"
}

post_check(){
  log "Verifying..."
  conda run -n "${ENV_NAME}" python - <<'PY'
import sys, platform
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
try:
  import torch
  print("Torch:", torch.__version__)
  print("CUDA available:", torch.cuda.is_available())
except Exception as e:
  print("Torch check:", e.__class__.__name__, e)
PY
  ok "Env ready. Activate: conda activate ${ENV_NAME}"
}

# ---------- Main ----------
main(){
  need_cmd uname; need_cmd bash
  ensure_miniforge
  init_shell
  ensure_mamba
  create_env_if_needed

  local CONDA_ONLY
  CONDA_ONLY="$(build_conda_only_yml)"
  log "Conda-only YML at: ${CONDA_ONLY}"

  apply_conda_only "${CONDA_ONLY}"
  install_pip_seq
  pin_python_again
  post_check
}

main "$@"
