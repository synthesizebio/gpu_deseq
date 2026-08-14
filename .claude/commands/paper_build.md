---
description: Recompile the cuDESeq2 whitepaper (paper/main.tex -> paper/main.pdf)
argument-hint: "[--quick|--figures|--clean]"
allowed-tools: Bash(scripts/paper_build.sh:*)
---
Run the paper build script and report the outcome (output path, page count, and
any unresolved references or errors):

!`scripts/paper_build.sh $ARGUMENTS`
