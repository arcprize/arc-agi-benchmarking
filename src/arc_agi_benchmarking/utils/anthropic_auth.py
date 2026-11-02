"""AWS workload identity configuration shared by Anthropic and preflight."""

import os

import boto3

ANTHROPIC_WIF_REQUIRED_ENV = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
)


def read_anthropic_wif_environment() -> dict[str, str] | None:
    """Select WIF explicitly, rejecting partial settings instead of using a key."""
    if not any(name in os.environ for name in ANTHROPIC_WIF_REQUIRED_ENV):
        return None

    settings = {
        name: os.environ.get(name, "").strip() for name in ANTHROPIC_WIF_REQUIRED_ENV
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(
            "Incomplete Anthropic WIF environment configuration. "
            f"Set non-empty values for: {', '.join(missing)}. "
            "To use an API key instead, unset all four Anthropic WIF variables."
        )
    return settings


def get_sts_web_identity_token() -> str:
    """Fetch a fresh AWS identity token whenever the Anthropic SDK refreshes."""
    sts = boto3.client("sts", region_name="us-east-1")
    response = sts.get_web_identity_token(
        Audience=["https://api.anthropic.com"],
        SigningAlgorithm="RS256",
        DurationSeconds=900,
    )
    return response["WebIdentityToken"]
