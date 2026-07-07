# Engineering Onboarding Guide

## Getting Access

New engineers request access to internal systems through the IT Service Portal.
Approval for production systems requires sign-off from your team lead and takes
1–2 business days. VPN access is provisioned automatically on your first day.

## Development Environment

We standardize on Python 3.12 and Node.js 20. Clone the monorepo and run
`make bootstrap` to install dependencies and Git hooks. The bootstrap script also
configures your local `.env` from the shared template in the vault.

## Code Review

Every change ships through a pull request with at least one approving review.
CI must be green before merge. Reviews are expected within one business day; ping
the `#eng-reviews` channel if a PR is blocking you.

## Deployment

Deploys are continuous: merging to `main` triggers a staging deploy automatically.
Production deploys are gated behind a manual approval in the pipeline and can only
be triggered by on-call engineers during business hours, except for hotfixes.
