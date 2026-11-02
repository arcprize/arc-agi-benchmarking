"""Check WIF selection, startup validation, and SDK token exchange offline."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import anthropic
import boto3
import httpx
import pytest
from anthropic import Anthropic, WorkloadIdentityCredentials
from anthropic.lib.credentials import TokenCache
from botocore.stub import Stubber

from arc_agi_benchmarking.adapters import anthropic as anthropic_adapter
from arc_agi_benchmarking.schemas import ModelConfig
from arc_agi_benchmarking.utils import anthropic_auth, preflight


@pytest.fixture(autouse=True)
def clear_auth_environment(monkeypatch):
    for name in (
        *anthropic_auth.ANTHROPIC_WIF_REQUIRED_ENV,
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def wif_environment(monkeypatch):
    settings = {
        "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
        "ANTHROPIC_ORGANIZATION_ID": "00000000-0000-0000-0000-000000000001",
        "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
        "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    return settings


@pytest.fixture
def model_config():
    # WIF configs can omit API-key metadata entirely.
    return ModelConfig(
        name="test-wif",
        model_name="test-model",
        provider="anthropic",
        pricing={"date": "2026-11-02", "input": 1.0, "output": 2.0},
        kwargs={"max_tokens": 16},
    )


@pytest.fixture
def adapter(model_config):
    instance = anthropic_adapter.AnthropicAdapter.__new__(
        anthropic_adapter.AnthropicAdapter
    )
    instance.model_config = model_config
    return instance


def test_wif_preflight_and_adapter_init_need_no_api_key_or_network(
    monkeypatch, wif_environment, model_config, tmp_path
):
    monkeypatch.setattr(preflight, "read_models_config", lambda _: model_config)
    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        lambda _: model_config,
    )
    sts = Mock(side_effect=AssertionError("No AWS call expected during startup"))
    monkeypatch.setattr(anthropic_auth.boto3, "client", sts)
    data_dir = tmp_path / "tasks"
    data_dir.mkdir()
    (data_dir / "task.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        )
    )

    report = preflight.run_preflight(
        model_config.name, str(data_dir), str(tmp_path / "outputs")
    )
    assert report.all_passed
    assert "WIF environment configuration found" in report.validations[1].message
    instance = anthropic_adapter.AnthropicAdapter(model_config.name)
    instance.client.close()
    sts.assert_not_called()


@pytest.mark.parametrize("missing", anthropic_auth.ANTHROPIC_WIF_REQUIRED_ENV)
@pytest.mark.parametrize("value", [None, "", "   "])
def test_partial_wif_fails_preflight_and_adapter_even_with_api_key(
    monkeypatch, wif_environment, adapter, missing, value
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-legacy-key")
    if value is None:
        monkeypatch.delenv(missing)
    else:
        monkeypatch.setenv(missing, value)

    report = preflight.validate_api_key("anthropic", "ANTHROPIC_API_KEY")
    assert not report.passed
    assert f"Set non-empty values for: {missing}" in report.details
    with pytest.raises(ValueError, match=f"Set non-empty values for: {missing}"):
        adapter.init_client()


@pytest.mark.parametrize("name", anthropic_auth.ANTHROPIC_WIF_REQUIRED_ENV)
def test_any_wif_variable_selects_wif(monkeypatch, adapter, name):
    monkeypatch.setenv(name, "configured")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-legacy-key")
    assert not preflight.validate_api_key("anthropic", "ANTHROPIC_API_KEY").passed
    with pytest.raises(ValueError, match="Incomplete Anthropic WIF environment"):
        adapter.init_client()


@pytest.mark.parametrize(
    "api_key_env", ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_NO_ZDR"]
)
def test_legacy_api_key_authentication(monkeypatch, adapter, api_key_env):
    adapter.model_config.api_key_env = api_key_env
    monkeypatch.setenv(api_key_env, "selected-legacy-key")
    assert preflight.validate_api_key("anthropic", api_key_env).passed
    factory = Mock()
    monkeypatch.setattr(anthropic_adapter.anthropic, "Anthropic", factory)
    adapter.init_client()
    factory.assert_called_once_with(api_key="selected-legacy-key", max_retries=0)


def test_without_wif_legacy_key_configuration_is_still_required(adapter, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-default-key")
    assert not preflight.validate_api_key("anthropic").passed
    with pytest.raises(ValueError, match="api_key_env must be configured"):
        adapter.init_client()


def test_without_wif_missing_custom_api_key_is_rejected(adapter, monkeypatch):
    adapter.model_config.api_key_env = "CUSTOM_ANTHROPIC_KEY"
    monkeypatch.delenv("CUSTOM_ANTHROPIC_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-default-key")
    assert not preflight.validate_api_key("anthropic", "CUSTOM_ANTHROPIC_KEY").passed
    with pytest.raises(ValueError, match="CUSTOM_ANTHROPIC_KEY not found"):
        adapter.init_client()


@pytest.mark.parametrize("provider", ["openrouter", "claudeagentsdk"])
def test_other_providers_keep_api_key_validation(monkeypatch, provider):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_partial")
    monkeypatch.delenv("OTHER_PROVIDER_KEY", raising=False)
    assert not preflight.validate_api_key(provider, "OTHER_PROVIDER_KEY").passed
    monkeypatch.setenv("OTHER_PROVIDER_KEY", "other-provider-key")
    assert preflight.validate_api_key(provider, "OTHER_PROVIDER_KEY").passed
    with pytest.raises(ValueError, match="api_key_env is required"):
        ModelConfig(
            name="other",
            model_name="test",
            provider=provider,
            pricing={"date": "2026-11-02", "input": 1.0, "output": 2.0},
        )


@pytest.mark.parametrize("workspace", ["wrkspc_test", "wrkspc_other"])
def test_wif_exchanges_caches_and_refreshes_with_fresh_aws_tokens(
    monkeypatch, wif_environment, adapter, workspace
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unused-auth-token")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", f" {workspace} ")
    sts = boto3.client(
        "sts",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    sts_factory = Mock(return_value=sts)
    monkeypatch.setattr(anthropic_auth.boto3, "client", sts_factory)
    exchanges = []
    message_headers = []
    clock = [time.time()]

    def respond(request):
        if request.url.path == "/v1/oauth/token":
            exchanges.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "access_token": f"access-token-{len(exchanges)}",
                    "token_type": "Bearer",
                    "expires_in": 600,
                },
            )
        assert request.url.path == "/v1/messages"
        message_headers.append(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
        Stubber(sts) as stub,
    ):
        for token in ("aws-jwt-1", "aws-jwt-2"):
            stub.add_response(
                "get_web_identity_token",
                {
                    "WebIdentityToken": token,
                    "Expiration": datetime.now(timezone.utc) + timedelta(seconds=900),
                },
                {
                    "Audience": ["https://api.anthropic.com"],
                    "SigningAlgorithm": "RS256",
                    "DurationSeconds": 900,
                },
            )
        monkeypatch.setattr(
            anthropic_adapter.anthropic,
            "WorkloadIdentityCredentials",
            lambda **kwargs: WorkloadIdentityCredentials(
                **kwargs, http_client=http_client
            ),
        )
        monkeypatch.setattr(
            anthropic_adapter.anthropic,
            "Anthropic",
            lambda **kwargs: Anthropic(
                **kwargs,
                http_client=http_client,
                _token_cache=TokenCache(
                    kwargs["credentials"], time_source=lambda: clock[0]
                ),
            ),
        )
        with adapter.init_client() as client:
            assert client.max_retries == 0
            sts_factory.assert_not_called()
            adapter.client = client
            for advance in (0, 0, 601):
                clock[0] += advance
                response = adapter.chat_completion(
                    [{"role": "user", "content": "test"}]
                )
                assert response.content[0].text == "ok"

        stub.assert_no_pending_responses()
        assert sts_factory.call_count == 2
        sts_factory.assert_called_with("sts", region_name="us-east-1")
        assert [exchange["assertion"] for exchange in exchanges] == [
            "aws-jwt-1",
            "aws-jwt-2",
        ]
        for exchange in exchanges:
            assert (
                exchange["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
            )
            assert (
                exchange["federation_rule_id"]
                == wif_environment["ANTHROPIC_FEDERATION_RULE_ID"]
            )
            assert (
                exchange["organization_id"]
                == wif_environment["ANTHROPIC_ORGANIZATION_ID"]
            )
            assert (
                exchange["service_account_id"]
                == wif_environment["ANTHROPIC_SERVICE_ACCOUNT_ID"]
            )
            assert exchange["workspace_id"] == workspace
        assert [headers["authorization"] for headers in message_headers] == [
            "Bearer access-token-1",
            "Bearer access-token-1",
            "Bearer access-token-2",
        ]
        assert all("x-api-key" not in headers for headers in message_headers)


def test_aws_failure_does_not_fall_back_to_api_key(
    monkeypatch, wif_environment, adapter
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-api-key")
    monkeypatch.setattr(
        anthropic_auth.boto3,
        "client",
        Mock(side_effect=RuntimeError("AWS unavailable")),
    )
    send = Mock(side_effect=AssertionError("No HTTP request expected"))
    with httpx.Client(transport=httpx.MockTransport(send)) as http_client:
        monkeypatch.setattr(
            anthropic_adapter.anthropic,
            "Anthropic",
            lambda **kwargs: Anthropic(**kwargs, http_client=http_client),
        )
        with adapter.init_client() as client:
            with pytest.raises(anthropic.APIConnectionError) as exc_info:
                client.messages.create(
                    model="test-model",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "test"}],
                )
            assert isinstance(exc_info.value.__cause__, RuntimeError)
            assert str(exc_info.value.__cause__) == "AWS unavailable"
        send.assert_not_called()
