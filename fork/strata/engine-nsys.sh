#!/bin/bash
# Strata's engine under nsys, as the "exe" of a server.py config (fork/strata/llama-server, the -nsys arm): a per-kernel trace of STRATA_NSYS_S seconds
# of decode, starting STRATA_NSYS_DELAY seconds after the engine's start. nsys runs in a session of its own and ends the engine itself after the window
# (the request under way fails, and the cell with it): the report it writes is out of reach of the cell's stop.
# the launch's own engine: component:strata-inkling lives under shapes/inkling2 (STRATA_NSYS_EXE from the stand-in), the rest is the main engine
case "${STRATA_NSYS_EXE:-component:strata}" in
    component:strata-inkling) eng=$(ls -d "$XDG_DATA_HOME"/strata/components/shapes/inkling2/engine/*/ | sort -V | tail -1); bin=strata-inkling ;;
    *) eng=$(ls -d "$XDG_DATA_HOME"/strata/components/engine/*/ | head -1); bin=strata ;;
esac
export LD_LIBRARY_PATH="$eng${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
setsid nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --delay="${STRATA_NSYS_DELAY:-50}" --duration="${STRATA_NSYS_S:-5}" \
    -o "${STRATA_NSYS_OUT:-/p/bw/data/nsys/strata}-%p" --force-overwrite=true "${eng}$bin" "$@" <&0 &
wait $!
