# Publishing wana 0.5.0

Packaging follows Wyra: Hatchling, a dynamic version in `wana/__init__.py`, a
console script, a tested wheel/sdist artifact, and Trusted Publishing on pushes to `main` (or published GitHub Releases). Creating files locally does not reserve the package name or publish it.

## One-time setup

1. Create/connect the intended GitHub repository, `sauloleite/wana`. If using
   another repository, update the URLs in `pyproject.toml` and publisher settings.
2. On PyPI, confirm that you control `wana` or that the name is available. An HTTP
   404 alone does not guarantee a name can be registered. For a new project,
   configure a pending publisher in the account's publishing settings.
3. Configure these exact publisher fields:

   | Field | Value |
   | --- | --- |
   | Project | `wana` |
   | Owner | `sauloleite` |
   | Repository | `wana` |
   | Workflow | `cicd.yml` |
   | Environment | `pypi-publish` |

4. Create the GitHub environment `pypi-publish`; configure release protections
   appropriate to the repository. No long-lived PyPI API token is needed.

See the official [pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [attestation documentation](https://docs.pypi.org/attestations/producing-attestations/).

## Release

1. Review the changes; run the development checks from README.
2. For new package contents, increment `__version__` in `wana/__init__.py` and
   update the changelog. This checkout prepares `0.5.0`; `0.1.0` is already published.
3. Commit and push to `main`. No tag or GitHub Release is required.
4. Wait for all CI platforms to pass. Local validation alone does not verify that matrix.
5. The workflow tests, builds, checks metadata, smoke-tests the wheel and uploads
   the same distribution bytes to PyPI with OIDC and attestations.
6. In a fresh environment, run `pip install wana==0.5.0` and `wana --version`.

Pull requests only test and build; they never publish. Published GitHub Releases
remain supported, with a tag matching the version (for example `v0.5.0`).
Publication jobs are serialized and `skip-existing: true` skips files already on
PyPI. A push with the same version does **not** update the installed package:
increment the version to distribute changes.

If publisher setup or name availability prevents upload, resolve that state
before retrying. If the GitHub environment restricts deployment branches, allow
`main` for this push-based flow. Do not overwrite an existing PyPI version.

The Node package is independent. See [node/README.md](../node/README.md) for
`@sauloleite/wana` and the manual `npm.yml` workflow. The local npm client is
currently unauthenticated, so no npm upload has been made.

## Exact phase-zero compatibility version

The manual `cicd.yml` input `bootstrap=true` builds version **0.0.1** from the
original skeleton commit `1f158af6b438f031fc933b12ae0583729f5f3dae`. It changes
only the package version and adds an explicit compatibility notice to the README.
This is a newly published compatibility version, not a claim that it was released
before 0.1.0. It has the original dependency-free check CLI; current scoring
features and bundled model weights belong to 0.5.0.

The workflow installs the exact 0.0.1 wheel and verifies `wana --version` before
uploading through the same configured Trusted Publisher. The bootstrap wheel was
also built and installed in a clean local environment. After the workflow exists
on `main`, run `gh workflow run cicd.yml -f bootstrap=true`. Verify the registry
installation afterwards with `pip install wana==0.0.1` in a fresh environment.
