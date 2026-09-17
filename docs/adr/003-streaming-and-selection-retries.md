# ADR 003: incremental scoring and verifiable selection retries

Status: implemented in 0.5.1.

The collecting Python API remains compatible. The new `iter_scored` generator
and the score CLI calculate one record at a time. Duplicate IDs still require an
identity set; no claim of constant total memory is made. Output is written to a
temporary file beside the destination, then replaced after success. Failure
preserves the previous file and removes the temporary file.

File checks replay training per matcher. This avoids retaining training text
while keeping evaluation indexes, identities and evidence in memory. In-memory
inputs can be one-shot iterators and are collected for replay. Selectors need
global ranks/strata/corpus statistics and the current run API returns complete
dataclasses. Their memory use remains a documented unfinished part of the plan.

A stateless fractional operation cannot both take 20% of each current input and
return the same output when repeated. Standalone CLI selection treats identical
parameters and package version on a verified prior selection artifact as a retry.
It copies the selected bytes, records reuse and links the prior manifest.
All direct previous artifact hashes must still verify. Different parameters,
versions or absent matching provenance use ordinary selection.
`--reselect` explicitly requests another reduction. In-memory selection keeps its
current-input semantics. A reused manifest is evidence of a retry, not a signature
or proof that the previous selection was optimal.
