# Attribution and provenance

BenStein1 directed and conducted this Pryon reverse-engineering/custom wake-word investigation, including its
numerical hypotheses, experiments, hardware testing, and the subsequent reconstruction/training work. Implementation
and analysis were developed with AI assistance. Independently implemented means reconstructed from observed binary
behavior; it does not mean no binary disassembly was used or that every mathematical primitive was invented here.

The Echo platform is [Gamer92000/echo-dot-assist](https://github.com/Gamer92000/echo-dot-assist), by Julian Imhof and
its contributors. Home Assistant integration, daemon infrastructure, device setup and the original project are their
work. Their MIT notice is retained in `licenses/echo-dot-assist-MIT.txt`. `native/support/json.*` and `hash.*` are copied
from that project. The math oracle and debugger improvements also passed through the reviewed fork contributions.
The Echo unlock/bootloader work belongs to the original authors credited by that project, including R0rt1z2.

The standalone core is an allowlisted export of the research sources. Portable helper functions were extracted
without bringing across private corpus loaders or machine defaults. Two prompt/dataset tests whose modules are
excluded were omitted; retained tests cover the selected numerical contracts. Paths to private assets are explicit
caller inputs. No original research Git history is included.

Kaldi, Ne10, OpenBLAS and Sun/FreeBSD msun were studied or ported where identified. The algorithm/implementation
attribution and full applicable notices are preserved in `THIRD_PARTY_NOTICES.md` and `licenses/`.

Amazon/Pryon/NemoRT names identify the system studied. Amazon did not author, endorse or supply this reconstruction.
No claim of permission to distribute Amazon firmware, model weights or derived model packages is made.
