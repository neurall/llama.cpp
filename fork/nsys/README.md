# nsys trace readers (decode, one card)

Trace: `python3 fork/run.py run onecard -b BUILD -onl 'p1-allk$'` writes `/p/bw/data/nsys/allk-PID.nsys-rep`;
`nsys export --type sqlite -o X.sqlite X.nsys-rep`. Strata's reference trace: `/p/bw/data/nsys/strata-1888937.sqlite`.

| script | prints |
|---|---|
| `overlap.py A.sqlite [B.sqlite ...]` | per token: wall, work kernels summed, card busy with work, work in parallel, time with no work running; top pairs of kernels that run together. Spin kernels (`k_moe_rows`, Strata's `wait_flag_ge`) are left out of "work" |
| `layers.py A.sqlite ...` | layer wall time by kind (delta-net, attention), wait between the delta net's end and the gated norm |
| `layer_one.py A.sqlite` | every kernel of one median delta-net layer: start, duration, stream id |
| `families.py OURS.sqlite STRATA.sqlite [N]` | per token time and calls by kernel name, both engines |

Stream ids of kernels launched from a CUDA graph are the graph executor's own, not our capture streams: read the plan with the `p1-plan` row
(`GGML_CUDA_STREAMS_DEBUG=1`, lines `CUDA0 streams:` in the server log) to see which stream a launch was captured on.
