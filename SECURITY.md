# Security Policy

## Supported versions

| Version | Supported |
| :-- | :-: |
| `master` / latest GHCR image | ✅ |
| Older images and model versions | ❌ (redeploy from `master`) |

## Reporting a vulnerability

> [!IMPORTANT]
> **Do not open a public issue, discussion or pull request for a security
> problem.**

Report it privately to a maintainer:

1. **Preferred:** if the repository is public, use GitHub's
   [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
   (*Security → Report a vulnerability*).
2. **Otherwise:** contact [@KrishKamra](https://github.com/KrishKamra) or
   [@amulyagavankar20-glitch](https://github.com/amulyagavankar20-glitch)
   directly through GitHub.

Please include:

- The affected component (API, training flow, Docker image, CI workflow, …) and
  the commit or image tag
- Steps to reproduce, or a proof of concept
- The impact you expect (data exposure, code execution, denial of service, …)

| Stage | Target time |
| :-- | :-- |
| Acknowledgement | 3 business days |
| Initial assessment | 7 business days |
| Fix or mitigation for critical issues | 30 days |

We will credit reporters in the release notes unless you ask us not to.

## Scope

**In scope**

- The FastAPI service (`src/api`) and its input validation
- Model bundle loading (`src/serving`), including deserialization
- Docker images, `docker-compose.yml` and the GitHub Actions workflows
- Dependency vulnerabilities that are exploitable in this project

**Out of scope**

- Model accuracy or clinical correctness (open a normal issue instead)
- Findings that require an already-compromised host or registry
- Vulnerabilities in the default local-only credentials in `.env.example`,
  which are documented as unsafe for shared environments

## Security model & hardening notes

> [!WARNING]
> **Model artifacts are pickles.** `bundle.joblib` is deserialised with
> `joblib`, which can execute code. Only load bundles produced by this
> project's pipeline from a registry or storage location you control. Never
> load a bundle from an untrusted source.

| Area | Current state | Deployment requirement |
| :-- | :-- | :-- |
| Authentication | None in the API | Put an API gateway, ingress or service mesh in front of it that enforces authN/Z |
| Transport | Plain HTTP | Terminate TLS at the gateway or ingress |
| Health data | Request bodies are not logged or stored | Keep it that way; review any new logging for PHI/PII |
| Container | Non-root user, slim base image, Trivy scan in CI | Pin image digests in production |
| Secrets | Read from environment / `.env` (git-ignored) | Use a secret manager; rotate the Compose defaults |
| Dependencies | Locked in `uv.lock`; Dependabot weekly | Merge security updates promptly |
| Input validation | Strict schema: bounds, enums, unknown fields rejected | — |

## Data handling

The repository contains a clinical dataset. Do not commit additional patient
data, credentials or real `.env` files. If sensitive data is committed by
mistake, report it as described above so that it can be purged from the
history.
