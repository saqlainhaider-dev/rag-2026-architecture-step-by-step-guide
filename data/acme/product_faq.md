# Acme Product FAQ

Last updated: 2026-02-01
Owner: Product
Access: public

## What is Acme Cloud?

Acme Cloud is a workspace platform for teams to manage projects, documents, and automated workflows. Plans include Free, Pro, and Enterprise.

## Pricing Overview

- **Free**: up to 3 users, 5 GB storage, community support
- **Pro**: $18 per user / month, 100 GB storage, priority email support
- **Enterprise**: custom pricing, SSO, audit logs, dedicated success manager

Annual Pro billing receives a **15% discount**.

## SSO and Security

SSO (SAML and OIDC) is available on **Enterprise** only. Pro customers can request early SSO access through their account manager; availability is not guaranteed.

All plans encrypt data in transit (TLS 1.2+) and at rest (AES-256).

## API Access

API keys are available on Pro and Enterprise. Rate limits:
- Pro: 60 requests / minute
- Enterprise: 600 requests / minute (higher limits on request)

API documentation lives at `https://docs.acme.example/api`.

## Data Export

Users can export workspace data as ZIP (files) plus JSON (metadata) from **Settings → Privacy → Export**. Exports complete within 24 hours for workspaces under 50 GB.
