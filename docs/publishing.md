# Publishing wana 0.1.0

Packaging follows Wyra: Hatchling, a dynamic version in `wana/__init__.py`, a
console script, a tested wheel/sdist artifact, and release-triggered Trusted
Publishing. Creating files locally does not reserve the package name or publish it.

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
2. Update the version and changelog, commit and push to the configured repository.
3. Wait for all CI platforms to pass. Local validation alone does not verify that matrix.
4. Publish a GitHub Release tagged `v0.1.0`, matching the package version.
5. The workflow tests, builds, checks metadata, smoke-tests the wheel and uploads
   the same distribution bytes to PyPI with OIDC and attestations.
6. In a fresh environment, run `pip install wana==0.1.0` and `wana --version`.

The publish job runs only on a published release. If publisher setup or name
availability prevents upload, resolve that state before another release attempt.
Do not overwrite an existing PyPI version. npm reservation and Node parity belong
to the later Node phase and are not part of this Python release.
