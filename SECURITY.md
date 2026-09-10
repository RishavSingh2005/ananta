# Security Policy

## Local Use

Keep the API bound to `127.0.0.1` for local use. Do not commit checkpoints,
API keys, `.env` files, raw private data, or generated logs.

## Public Deployment Requirements

Before binding to `0.0.0.0` or placing Ananta behind a public domain:

1. Set a long random `ANANTA_API_KEY` outside the repository.
2. Put the service behind an HTTPS reverse proxy.
3. Set `ANANTA_CORS_ORIGINS` to explicit trusted origins.
4. Configure rate limits, request timeouts, and process memory limits.
5. Run the service as a non-administrator account.
6. Add monitoring and rotate logs without storing sensitive prompts by default.
7. Keep the model checkpoint immutable and maintain rollback copies.

Keep `ANANTA_ENABLE_CODE_RUNNER=0` for public deployments. The local runner is
an opt-in convenience with a timeout, output cap, temporary directory, and no
shell invocation; it is not a complete operating-system sandbox. Use a real
container or VM isolation layer before executing untrusted public code.

The built-in rate limiter is process-local. Multi-worker or multi-instance
deployments need a shared gateway or Redis-backed limiter.

## Reporting

Do not publish secrets or private prompts in an issue. Report reproducible
security problems with affected endpoint, version, reproduction steps, and
impact. Remove credentials and personal data from the report.

Teacher API keys are temporary credentials. Set `OPENAI_API_KEY` only in the
current shell, never in source files or committed `.env` files, and remove it
after collecting approved candidate data. Do not send private or unauthorized
copyrighted material to a teacher API.