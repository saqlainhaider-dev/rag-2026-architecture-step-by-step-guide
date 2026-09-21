# Acme On-Call Runbook

Last updated: 2026-03-01
Owner: SRE
Access: internal

## Severity Levels

- **SEV-1**: Full outage or data-loss risk. Page primary + secondary immediately.
- **SEV-2**: Major feature degraded for many users. Page primary within 5 minutes.
- **SEV-3**: Limited impact or workaround exists. Create ticket; respond during business hours.

## First Response Checklist

1. Acknowledge the page in PagerDuty within **5 minutes**
2. Join the `#incident-bridge` Slack channel
3. Declare severity and incident commander
4. Check status of API gateway, auth service, and Postgres primary
5. Post a customer-facing status update within **15 minutes** for SEV-1/SEV-2

## Rollback Procedure

If a bad deploy is suspected:
1. Identify the release tag from the deploy dashboard
2. Run `acmectl rollback --service <name> --to previous`
3. Verify error rate and latency on the Grafana **Service Health** board
4. Keep the incident open until metrics are stable for 20 minutes

## Escalation

Escalate to the platform team if:
- Auth outages last longer than 10 minutes
- Postgres failover does not complete automatically
- Customer data integrity is in question

Escalation contact: platform-oncall@acme.example
