# @sauloleite/wana

Node port of Wana's deterministic contamination checks and manifest verification.
This package lives in the same repository as the Python package, with independent
packaging. Requires Node 20+. No scoring model, training or IFD in the Node package.

After the first npm publication:

```sh
npm install @sauloleite/wana
npx wana check train.jsonl --eval valid.jsonl -o contamination.json
npx wana verify selected/manifest.json
```

```js
import {readJsonl, checkContamination, verifyManifest} from '@sauloleite/wana';
const report = checkContamination(readJsonl('train.jsonl'), readJsonl('valid.jsonl'));
console.log(report.ok);
```

Options: `--ngram 13`, `--shingle 5`, `--threshold 0.8`, repeated
`--ignore-template TEXT`, `--fail-on EXACT,NEAR`. Exit codes: 0 pass, 1 policy
violation or hash mismatch, 2 invalid input/I/O. Inputs may be OpenAI chat,
Alpaca or ShareGPT JSONL; gzip is supported. Use Python for xz/bzip2.
NEAR is approximate LSH with exact Jaccard verification. Short records can
produce no hits. No semantic embedding detection is included in this port.

`serializeReport` produces the shared golden report byte-for-byte. Extra
multilingual and near-match fixtures are compared structurally. Hash verification
checks all directly referenced artifacts and the parent file, not authenticity.
Legacy relative paths are relative to the process working directory; new Python
pipeline manifests store absolute paths.

Development: `npm install`, `npm test`, `npm pack --dry-run`. Tests use fixtures
from `../tests/fixtures` and must run from this repository checkout.

Publication: first establish ownership of the `@sauloleite` npm scope and publish
using your npm account. Configure npm Trusted Publishing for repository
`sauloleite/wana`, workflow `npm.yml`, environment `npm-publish` before invoking
that workflow. There is no npm token committed to this repository.

The workflow publishes with provenance, which requires a public source repository
and a GitHub-hosted runner. If the repository is private, this workflow cannot
produce npm provenance. See the official [npm prerequisites](https://docs.npmjs.com/generating-provenance-statements/#prerequisites).
