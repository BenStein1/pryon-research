# Third-party notices

The proposed license for BenStein1's original release work is MIT (`LICENSE`). These terms do not supersede the notices
below or grant rights to excluded firmware, model weights, candidate packages or recordings.

- **echo-dot-assist:** Copyright (c) 2026 Julian Imhof; MIT. The complete notice is in
  `licenses/echo-dot-assist-MIT.txt`. It covers upstream support code in `native/support/json.*` and `hash.*`.
- **Ne10:** `pryon_lfbe.c` reimplements Ne10's level-1 real FFT operation order. Source attribution is
  Copyright 2014–16 ARM Limited and Contributors. The complete upstream Copyright 2012–16 notice, BSD-3-Clause
  conditions and disclaimer are in `licenses/Ne10-LICENSE.txt` and must accompany distributions.
- **Kaldi:** the frontend follows its window, power and mel-bank pipeline, with observed changes and operation order.
  The complete upstream legal notice/Apache-2.0 text is retained in `licenses/Kaldi-COPYING.txt`. That upstream notice
  describes additional Kaldi components; their mention does not mean they are incorporated here.
- **Sun/FreeBSD msun:** `pryon_lfbe.c` ports logf and retains the 1993 SunPro notice. `native/pryon_qat.c` ports
  exp/log/log1p and retains the applicable 2004 Sun notice: permission to use, copy, modify and distribute is freely
  granted provided the notice is preserved. These notices remain in the source and apply to standalone binaries too.
- **OpenBLAS:** the ARMv7 sdot accumulation order was observed and independently expressed. No OpenBLAS source file
  is copied into this candidate.

License texts were retained from the official projects and the already-reviewed Echo research contribution. This is
an engineering asset/provenance review, not a determination of rights to redistribute any excluded vendor asset.
Python dependencies remain separate installations with their own licenses; no dependency wheels are vendored.
