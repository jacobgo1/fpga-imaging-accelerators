# Tab completion for main.py. Source it, then add that line to ~/.bashrc:
#
#     source /path/to/fpga-imaging-accelerators/tabcompletion.sh
#
# It gives you a `main` command that works from any directory, and completes
# stages, flags and the values they accept:
#
#     main <TAB>                  doctor kernels parts native csim csynth ...
#     main csynth --<TAB>         --part --kernel --clock-ns --dry-run ...
#     main csynth --part <TAB>    zcu104 zybo-z7-20 arty-a7-35t ...
#     main csim --kernel <TAB>    the kernels present in src/hls/
#
# Every list is read from main.py and the config files when you press TAB, so
# adding a kernel, a board or a flag needs no change here.
#
# Works in Git Bash, WSL and Linux. For zsh, run
# `autoload -U +X bashcompinit && bashcompinit` before sourcing.

_MAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) _MAIN_PYTHON=python ;;
    *)                    _MAIN_PYTHON=python3 ;;
esac

main() { "$_MAIN_PYTHON" "$_MAIN_ROOT/main.py" "$@"; }

# Flags whose next word is a value, not a stage.
_MAIN_VALUE_FLAGS="--kernel --config --part --clock-ns --hls-tool --vivado-tool --cxx"

_main_ask() { "$_MAIN_PYTHON" "$_MAIN_ROOT/main.py" "$@" 2>/dev/null; }

_main_stages() {
    _main_ask --help | grep -oE '\{[a-z,]+\}' | head -1 | tr -d '{}' | tr ',' ' '
}

_main_flags() {
    _main_ask --help | grep -oE '\-\-[a-z-]+' | sort -u | tr '\n' ' '
}

# Field 1 of `main.py parts` is the board name, field 2 the exact part.
_main_parts() {
    _main_ask parts | awk -v field="$1" '/^  /{print $field}' | awk '!seen[$0]++'
}

# Kernel names are the unindented lines of `main.py kernels`.
_main_kernels() { _main_ask kernels | awk '!/^ /{print $1}'; }

_main_completion() {
    local current previous candidates word index skip stage
    current="${COMP_WORDS[COMP_CWORD]}"
    previous="${COMP_WORDS[COMP_CWORD-1]}"

    case "$previous" in
        --part)        if [[ "$current" == xc* ]]; then
                           candidates="$(_main_parts 2)"   # typing an exact part
                       else
                           candidates="$(_main_parts 1)"   # board names
                       fi ;;
        --kernel)      candidates="$(_main_kernels)" ;;
        --cxx)         candidates="g++ clang++" ;;
        --hls-tool)    candidates="vitis-run vitis_hls" ;;
        --vivado-tool) candidates="vivado" ;;
        --config)      COMPREPLY=($(compgen -f -- "$current")) ; return ;;
        --clock-ns)    return ;;                    # a number; nothing to suggest
        *)
            # Which stage, if any, has been given already?
            skip=0
            stage=""
            for ((index = 1; index < COMP_CWORD; index++)); do
                word="${COMP_WORDS[index]}"
                if [ "$skip" = 1 ]; then
                    skip=0
                elif [[ " $_MAIN_VALUE_FLAGS " == *" $word "* ]]; then
                    skip=1
                elif [[ "$word" != -* ]]; then
                    stage="$word"
                fi
            done
            if [ -z "$stage" ]; then
                candidates="$(_main_stages)"
            else
                candidates="$(_main_flags)"
                # Do not offer a flag that is already on the line.
                for ((index = 1; index < COMP_CWORD; index++)); do
                    case "${COMP_WORDS[index]}" in
                        --*) candidates="$(printf '%s\n' $candidates |
                                           grep -vx -- "${COMP_WORDS[index]}")" ;;
                    esac
                done
            fi
            ;;
    esac

    COMPREPLY=($(compgen -W "$candidates" -- "$current"))
}

complete -o default -F _main_completion main
