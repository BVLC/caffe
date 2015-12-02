#!/bin/bash

# grunt-cli
# http://gruntjs.com/
#
# Copyright (c) 2012 Tyler Kellen, contributors
# Licensed under the MIT license.
# https://github.com/gruntjs/grunt/blob/master/LICENSE-MIT

# Usage:
#
# To enable bash <tab> completion for grunt, add the following line (minus the
# leading #, which is the bash comment character) to your ~/.bashrc file:
#
# eval "$(grunt --completion=bash)"

# Search the current directory and all parent directories for a gruntfile.
function _grunt_gruntfile() {
  local curpath="$PWD"
  while [[ "$curpath" ]]; do
    for gruntfile in "$curpath/"{G,g}runtfile.{js,coffee}; do
      if [[ -e "$gruntfile" ]]; then
        echo "$gruntfile"
        return
      fi
    done
    curpath="${curpath%/*}"
  done
  return 1
}

# Enable bash autocompletion.
function _grunt_completions() {
  # The currently-being-completed word.
  local cur="${COMP_WORDS[COMP_CWORD]}"
  # The current gruntfile, if it exists.
  local gruntfile="$(_grunt_gruntfile)"
  # The current grunt version, available tasks, options, etc.
  local gruntinfo="$(grunt --version --verbose 2>/dev/null)"
  # Options and tasks.
  local opts="$(echo "$gruntinfo" | awk '/Available options: / {$1=$2=""; print $0}')"
  local compls="$(echo "$gruntinfo" | awk '/Available tasks: / {$1=$2=""; print $0}')"
  # Only add -- or - options if the user has started typing -
  [[ "$cur" == -* ]] && compls="$compls $opts"
  # Tell complete what stuff to show.
  COMPREPLY=($(compgen -W "$compls" -- "$cur"))
}

complete -o default -F _grunt_completions grunt
