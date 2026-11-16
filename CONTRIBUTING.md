# Contributing to climpred

See the [contribution guide](https://climpred.readthedocs.io/en/latest/contributing.html)
and the [Code of Conduct](CODE_OF_CONDUCT.md).

Using AI coding agents? Read the
[PR checklist](https://climpred.readthedocs.io/en/latest/contributing.html#ai-contributions)
([discussion #972](https://github.com/pangeo-data/climpred/discussions/972)). In short: issue with a use case first, understand each code
block yourself, write the gist yourself, test locally, few PRs at a time.

## AI policy

Based on the "standard" preset of [aipolicy.1mb.dev](https://aipolicy.1mb.dev/?preset=standard).

- **Accountability**: you are responsible for every line you submit, however it was
  produced. Review and verify all AI output before committing.
- **Usage**: AI tools are permitted. The normal review process applies.
- **Review**: AI-generated code is reviewed by a human maintainer before merging.
  Keep changes atomic; you must be able to explain any line on request.
- **Disclosure**: mark AI involvement in commit messages with a trailer, e.g.
  `Assisted-by: <tool name>` or `Co-authored-by: <tool> <email>`.
- **Prohibited**:
  - submitting AI output you have not read and understood
  - using AI to circumvent tests or CI checks
  - feeding reviewer feedback back into AI without understanding it first
  - AI-generated PR gists: write the gist yourself (the prompt may follow it)
  - AI-generated review comments on other contributors' PRs
- **CI/CD**: AI tools may be used in CI for code analysis and test generation.
- **Licensing**: AI-assisted contributions must be compatible with the MIT license,
  and you must be able to certify them under it.
- **Enforcement**: maintainers may close AI-generated PRs that do not meet these
  standards without detailed feedback.

This policy will evolve with tooling; contributions to it are welcome.
