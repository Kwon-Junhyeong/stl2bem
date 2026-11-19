#!/bin/bash
# Build figures: bash docs/figures/build.sh fig_sphere_field [fig_... ...]   (all fig_*.tex without arguments)
# Needs XeLaTeX (TeX Live or MiKTeX) and pdftoppm. Output: out/<name>.pdf and out/<name>.png (300 dpi);
# the PNG is copied to docs/img/ for the README.
set -u
cd "$(dirname "$0")"
mkdir -p out
names=("$@")
[ ${#names[@]} -eq 0 ] && names=($(ls fig_*.tex | sed 's/\.tex$//'))
for name in "${names[@]}"; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=out "$name.tex" > "out/$name.stdout" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$name] FAILED"; grep -n -A6 '^!' "out/$name.log" | head -20; continue
  fi
  pdftoppm -r 300 -png -singlefile "out/$name.pdf" "out/$name"
  cp "out/$name.png" "../img/$name.png"
  echo "[$name] ok, warnings: $(grep -c -E 'Overfull|Missing character' "out/$name.log")"
done
