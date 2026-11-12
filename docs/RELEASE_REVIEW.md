# Public-release boundary and approval

This source-only release was prepared and approved separately from the private research repository. The reviewed
initial snapshot `88487e5` is published at https://github.com/BenStein1/pryon-research with its MIT license and retained
third-party notices. Its original 58-file inventory remains available at that commit. The private repository's
visibility and historical commits remain unchanged. Subsequent publication-status documentation changes retain the
same source-only boundary; the current inventory reflects those files.

## Included

- Independently written frontend, quantized streaming/LSTM/decoder reconstruction and float ONNX translation tools.
- Recording-level head QAT, constrained dense-layer studies and source-preserving quantized export guards.
- Portable C trainer, msun-compatible math and its deterministic hash oracle.
- Read-only, hash-guarded QEMU/GDB capture tools and an explicit conv1 boundary comparator.
- Synthetic ONNX fixtures/signals/generated frontend vectors, tests and a firmware-free demonstration.
- Curated scalar results, rejected hypotheses, architecture diagrams, attribution, build recipes and complete notices.

## Excluded

No vendor executables/shared libraries, weights, learned arrays, ONNX model files, generated candidate models,
recordings, private datasets, runtime capture tensors, raw device logs, deployment/rollback bundles, private device
identifiers, local network addresses, machine-specific home paths, credentials, private Git objects, or unreviewed
working notes are exported. No private project README/handoff/agent instructions are copied verbatim.

The process-patching native extractor (including its instruction fingerprint) stays in the separately reviewed Echo
fork; this standalone candidate does not duplicate it. Debugger offsets and hashes are compatibility observations,
not shipped vendor binaries. Public golden vectors are generated regression data and explicitly not stock oracles.

## Review method

The private export tooling uses a filename allowlist; no recursive copy of the private tree, Git directory or artifact
folders occurs. Every publication path has a role/provenance entry and SHA-256 in `RELEASE_CONTENTS.json` (the manifest
itself is the one self-hash exception). A separate private source map records source hashes and transformations.
Tests run from the candidate with its own virtual environment, so private modules cannot silently satisfy imports.

Review includes text/file-type checks, machine path/address and credential-pattern checks, a local credential scanner,
all reachable historical blobs, initial-root ancestry, no-remotes checks, and a `git archive` inspection. Scanner
results have finite detection coverage; they are evidence, not a promise that arbitrary confidential prose can be
recognized automatically. The curated documents and source selection were reviewed separately from pattern scans.
See VALIDATION.md for the exact completed checks and remaining limits.

## Approval and future changes

The owner approved the reviewed source-only snapshot and publication as `BenStein1/pryon-research`. The original
private repository must remain private; its Git history and artifacts must not be imported into this release.
New artifacts require their own review before publication. No private artifacts belong in a release attachment.
