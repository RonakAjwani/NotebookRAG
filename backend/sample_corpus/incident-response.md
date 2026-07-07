# Incident Response Runbook

## Severity Levels

Incidents are classified SEV1 through SEV3. A SEV1 is a full outage or data-loss
event affecting all customers. A SEV2 is a major degradation affecting a subset of
customers or a critical feature. A SEV3 is a minor issue with a workaround.

## Declaring an Incident

Anyone can declare an incident in the `#incidents` channel using the `/incident`
command. Declaring an incident pages the current on-call engineer and opens a
dedicated incident channel. For SEV1 and SEV2, an incident commander is assigned.

## Roles

The incident commander coordinates the response and is the single decision-maker.
The communications lead posts status updates to the status page every 30 minutes
for SEV1s. The operations lead drives the technical investigation and mitigation.

## Postmortems

Every SEV1 and SEV2 requires a blameless postmortem within five business days.
The postmortem documents the timeline, root cause, and action items. Action items
are tracked to completion and reviewed in the weekly reliability meeting.

## On-Call

On-call rotations are weekly, handed off every Monday at 10:00. The on-call
engineer must acknowledge pages within 15 minutes. If unacknowledged, the page
escalates to the secondary on-call and then to the engineering manager.
