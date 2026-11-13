"""Auth middleware re-export for coding-analysis.

This module used to be a 174-line FORK of ``coding_client.auth.middleware``.
It was a stale copy, not a divergent implementation: its ``PUBLIC_PATHS`` was
byte-identical to the SDK's, its trusted-network parsing duplicated
``coding_client.auth._trusted_networks``, and its dispatch was the SDK's 1.9.0
dispatch with the comments rewritten.

Keeping the copy had a concrete cost. The SDK gained licence introspection in
1.10.0 (``AuthMiddleware`` reads ``Authorization: Bearer <licence>`` and
verifies it against the licences service); the fork did not, so this service had
NO working credential other than a bare, forgeable ``X-Tenant-Id`` header.
Measured live 2026-08-12 on the running pod: a valid licence Bearer that
returned 200 on coding-ai returned **401** here, while a forged
``X-Tenant-Id`` + ``X-User-Roles: admin`` returned **200**. That combination is
why ``TRUST_TENANT_HEADER=false`` could not be set on coding-analysis — with no
Bearer path the flag is a 401-everything switch rather than a hardening.
Consuming the SDK fixes that at the source instead of porting introspection
into a second implementation nobody would keep in step.

⚠️ HISTORY THAT MUST NOT BE LOST WITH THE FORK — the reason ``TRUSTED_CIDRS``
defaults to EMPTY. It previously defaulted to the RFC1918 ranges including
``10.0.0.0/8``, which contains the cluster pod CIDR ``10.42.0.0/16``, so any pod
in any namespace that could reach this service read tenant data with no
credential at all. **That is what leaked the code graphs on 2026-07-29.** A CIDR
here is a network position, not a credential — never widen it to cover the pod
CIDR. The SDK enforces the same empty default
(``coding_client.auth._trusted_networks.load_trusted_networks``) and logs a
warning at import if it is ever set.

Names are re-exported because this service's tests and callers import them from
here. ``_is_trusted_network`` is the SDK's ``is_trusted_network``, which reads
``TRUSTED_NETWORKS`` from its own module at CALL time, so monkeypatching
``coding_client.auth._trusted_networks.TRUSTED_NETWORKS`` works as it does in
the SDK's own suite.
"""

from coding_client.auth._trusted_networks import TRUSTED_NETWORKS, is_trusted_network
from coding_client.auth.middleware import PUBLIC_PATHS, AuthMiddleware

# Back-compat alias: this module exposed a private-by-name helper and the tests
# import it under that name.
_is_trusted_network = is_trusted_network

__all__ = [
    "PUBLIC_PATHS",
    "TRUSTED_NETWORKS",
    "AuthMiddleware",
    "_is_trusted_network",
]
