#!/bin/bash

objcopy -I binary -B i386 -O elf64-x86-64 $1 $2

sym="_binary_$(echo $1 | sed 's|[./]|\_|g')"
name="_$(basename ${1%.*})"

if [[ $name != "_cl"* ]]; then
  name="_cl${name}"
fi

for postfix in '_start' '_end' '_size'; do
  objcopy --redefine-sym "${sym}${postfix}"="${name}${postfix}" $2
done
