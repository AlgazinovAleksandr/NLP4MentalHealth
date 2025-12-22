from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests
from jose import jwt

from fastapi_app.config.config import settings


@dataclass(frozen=True)
class AuthContext:
    subject: str
    token: dict[str, Any]

    def has_role(self, role: str) -> bool:
        roles: set[str] = set()

        realm_access = self.token.get("realm_access") or {}
        roles.update(realm_access.get("roles") or [])

        resource_access = self.token.get("resource_access") or {}
        client_access = resource_access.get(settings.keycloak_client_id) or {}
        roles.update(client_access.get("roles") or [])

        return role in roles


class KeycloakJwksCache:
    def __init__(self) -> None:
        self._jwks: Optional[dict[str, Any]] = None

    def get_jwks(self) -> dict[str, Any]:
        if self._jwks is not None:
            return self._jwks

        url = (
            f"{settings.keycloak_internal_url.rstrip('/')}/realms/{settings.keycloak_realm}"
            "/protocol/openid-connect/certs"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        self._jwks = resp.json()
        return self._jwks


_jwks_cache = KeycloakJwksCache()


def _get_bearer_token(authorization: str) -> str:
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise ValueError("Invalid Authorization header; expected 'Bearer <token>'")
    return parts[1].strip()


def validate_jwt(authorization_header: str) -> AuthContext:
    token_str = _get_bearer_token(authorization_header)
    jwks = _jwks_cache.get_jwks()

    decoded = jwt.decode(
        token_str,
        jwks,
        options={"verify_aud": False},
        algorithms=["RS256", "RS384", "RS512"],
    )

    sub = decoded.get("sub")
    if not sub:
        raise ValueError("JWT missing 'sub'")

    return AuthContext(subject=sub, token=decoded)
