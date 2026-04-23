from __future__ import annotations

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_app.auth.auth import AuthContext, validate_jwt
from fastapi_app.config.config import settings

bearer_scheme = HTTPBearer(auto_error=False, scheme_name="bearerAuth")


def _keycloak_authorize_url() -> str:
    return (
        f"{settings.keycloak_public_url.rstrip('/')}/realms/{settings.keycloak_realm}"
        "/protocol/openid-connect/auth"
    )


def _build_login_redirect(request: Request) -> str:
    redirect_uri = settings.keycloak_redirect_uri
    if not redirect_uri:
        redirect_uri = str(request.base_url).rstrip("/") + "/"

    return (
        f"{_keycloak_authorize_url()}"
        f"?client_id={settings.keycloak_client_id}"
        "&response_type=code"
        "&scope=openid"
        f"&redirect_uri={redirect_uri}"
    )


def require_auth(request: Request, creds: HTTPAuthorizationCredentials | None = Security(bearer_scheme)) -> AuthContext:
    if creds is None or not creds.credentials:
        location = _build_login_redirect(request)
        raise HTTPException(status_code=307, detail="Redirect to Keycloak", headers={"Location": location})

    try:
        return validate_jwt(f"Bearer {creds.credentials}")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_admin(ctx: AuthContext = Security(require_auth)) -> AuthContext:
    if not ctx.has_role("admin"):
        raise HTTPException(status_code=403, detail="Forbidden")
    return ctx
