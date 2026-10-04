# README table, new tests (work in progress)

Filled by readme-update.py from the run log; it goes back into the README when the runs are complete.

Speed relative to stock llama.cpp, sorted by gain. t/s = generation speed (tokens/s), pp = prompt-processing speed; each speed cell shows stock on the top line and ours below, each gain cell the gain of ours over stock for t/s on top and pp below. short4 = four short game prompts (`game4`), long4 = four edit instructions over one long source file (`edit4`); each cell shows the best run of stock and the best run of ours, with the learned state kept between runs as in normal use. *Italic* values come from earlier tests or builds (b11707 where nothing newer exists) and will be replaced; `-` means not yet measured.

| model | hardware | in VRAM | short4 t/s<br>stock<br>ours | short4 pp<br>stock<br>ours | long4 t/s<br>stock<br>ours | long4 pp<br>stock<br>ours | short4 gain<br>t/s<br>pp | long4 gain<br>t/s<br>pp | build |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Flash-<br>RL-IQ3_XXS | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 36% | *4.4*<br>*11.4* | *3*<br>*4* | *4.2*<br>*9.1* | *156*<br>*112* | *2.6x*<br>*1.2x* | *2.3x*<br>*0.8x↓* | b11707 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 45% | *12.5*<br>*27.5* | *16*<br>*16* | - | - | *2.2x*<br>*1.0x* | - | b11707 |
| Qwen3.6-35B-A3B-<br>GSQ-hybrid | 4060 8G 4/8<br>8945HS 32G 48G/s | 73%<br>slow CPU | *31.6*<br>*60.0* | *41.2*<br>*100.9* | - | - | *1.9x*<br>*2.6x* | - | b12030 |
| Qwen3.6-35B-A3B-<br>GSQ-hybrid | 3600 A520 64G | 0%<br>CPU only | *6.3*<br>*11.1* | *10.1*<br>*34.1* | - | - | *1.8x*<br>*3.7x* | - | b12040 |
| Qwen3.8-Flash-<br>Next-UD-IQ4_XS | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 55% | 31.1<br>**51.6** | 17<br>23 | 30.9<br>**49.1** | 220<br>605 | 1.7x<br>1.3x | 1.6x<br>2.7x | b12193 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 44%<br>2 of 4 | ~17<br>30.4 | -<br>- | - | - | ~1.7x<br>- | - | b11707+ |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 66%<br>3 of 4 | *20.1*<br>*32.6* | -<br>- | - | - | *1.6x*<br>- | - | b11707+ |
| Qwen3.8-27B-MTP-Q5_K_M | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | fits | *78.3*<br>*77.5* | *41*<br>*40* | *31.3*<br>*51.1* | *599*<br>*792* | *1.0x*<br>*1.0x* | *1.6x*<br>*1.3x* | b11649 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 22%<br>1 of 4 | ~15<br>22.9 | -<br>- | - | - | ~1.6x<br>- | - | b11707+ |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 88%<br>second run, -t 16 | *24.7*<br>*33.8* | -<br>- | - | - | *1.4x*<br>- | - | b11707+ |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ3_S | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 58% | 43.4<br>**55.2** | 29<br>32 | 43.1<br>**53.2** | 380<br>687 | 1.3x<br>1.1x | 1.2x<br>1.8x | b12193 |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ1_M | 4060 8G 4/8<br>8945HS 32G 48G/s | 13% | *13.5*<br>*17.3* | *18.3*<br>*21.6* | - | - | *1.3x*<br>*1.2x* | - | b12030 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 88%<br>second run | *24.7*<br>*31.5* | -<br>- | - | - | *1.3x*<br>- | - | b11707+ |
| Qwen3.8-27B-IQ4_NL | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | fits<br>dense | 45.1<br>45.0 | 28<br>28 | 44.3<br>44.2 | 1756<br>1749 | 1.0x<br>1.0x | 1.0x<br>1.0x | b12193 |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ1_M | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 87% | 68.4<br>67.2 | 39<br>39 | 65.4<br>65.4 | 1209<br>1329 | 1.0x<br>1.0x | 1.0x<br>1.1x | b12193 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 88%<br>first run | 24.7<br>25.2 | -<br>- | - | - | 1.0x<br>- | - | b11707+ |
| any model that fits | any | 100% | same<br>same | same<br>same | - | - | 1.0x<br>1.0x | - | any |
| GLM-5.3-Flash-<br>GSQ-RCO-3.5bit | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 35% | 8.1<br>**12.2** | 5<br>5 | - | - | 1.5x<br>1.0x | - | b12209 |
| Qwen3.8-27B-GSQ-<br>RCO-IQ3_S-mtp | 3600 A520 64G | 0%<br>CPU only | *1.7*<br>*1.6* | *5.6*<br>*5.8* | - | - | *0.9x↓*<br>*1.0x* | - | b12030 |


