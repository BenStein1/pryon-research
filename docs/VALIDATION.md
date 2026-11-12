# Release-candidate validation

The checks below were completed during this release-preparation pass, on the candidate itself unless stated otherwise.

| Check | Result | Boundary |
|---|---|---|
| Python suite in candidate's own Python 3.12.8 environment | 32 tests passed | Synthetic fixtures only; no private source modules imported |
| Portable C math oracle | FNV-1a `96594667583f1fd3` passed | Fixed generated sequence, not all real-valued inputs |
| Public demo | Loss 1.4120500854 → 0.0365353360; one constructed accept | Synthetic features/probabilities; no vendor encoder or speech accuracy |
| Host portable trainer build | Passed | C compiler, no proprietary model/library |
| ARM trainer cross-build | Passed with NDK r21e, API 24 | Build only, not fresh training on device |
| Retained normalized conv1 comparison | 828/828 frames, 64 bins, max error 0 | Read-only reanalysis of existing genuine-runtime data, not fresh stock capture |
| Actual Dot 3 math-only test, daemon user | Reconstructed math and device libm both `96594667583f1fd3` | Protected-file SHA-1 before/after checks unchanged; no service restart; test files removed |
| Original private research unit harness | 144 tests; all three stages passed | Original environment, kept private; not substituted for isolated release tests |
| Combined Echo fork main | Host/Dot 3 daemon/extractor/trainer builds; enrollment/state/UI/package/math, artifact-copy and wake-word-removal tests passed | Host workflows use stand-ins; no live enrollment during publication pass |
| Combined Echo reconstruction tools | Five unittest and four standalone checks passed | Synthetic contracts and wire protocol |
| Source and history review | Filename allowlist, UTF-8/type/path/address checks; Gitleaks v8.30.1 | Exact reviewed tree and all reachable commit history; finite scanner coverage |

The hardware installer/confirmation/recovery and earlier web enrollment results described in EVIDENCE.md are prior
experiments, not reruns of the full live microphone workflow in this publication pass. Private native-model
integration through the separate runner was not dispatched here. No new claim about that gate is made.

No fresh held-out speaker/phrase/room corpus was collected. There is no production false-accept-per-hour benchmark,
no universal phrase-quality result, and no assurance for different firmware or Echo hardware. Public CI configuration
is supplied, but no hosted CI result is claimed for this unpublished repository.

This candidate has one new root commit with a no-reply identity, no inherited private commits or publication remote.
Its exact file inventory is RELEASE_CONTENTS.json. Publication and the proposed original-code MIT license await
owner approval; generated vendor-derived models remain excluded regardless of that approval.
