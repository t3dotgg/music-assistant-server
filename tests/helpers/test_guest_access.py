"""Tests for the guest access helpers."""

from __future__ import annotations

from ipaddress import ip_address
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import UserRole
from music_assistant_models.errors import InvalidDataError

from music_assistant.helpers import guest_access


def _create_mock_mass() -> MagicMock:
    """Create a mock MusicAssistant with a mocked auth controller."""
    mass = MagicMock()
    auth = mass.webserver.auth
    auth.get_user_by_username = AsyncMock(return_value=None)
    auth.create_user = AsyncMock()
    auth.get_active_join_code = AsyncMock(return_value=None)
    auth.generate_join_code = AsyncMock()
    auth.revoke_join_codes = AsyncMock(return_value=0)
    auth.revoke_tokens_for_user = AsyncMock(return_value=0)
    mass.webserver.external_url = None
    return mass


async def test_get_or_create_guest_user_returns_existing() -> None:
    """An existing guest user is returned without creating a new one."""
    mass = _create_mock_mass()
    existing_user = MagicMock(role=UserRole.GUEST)
    mass.webserver.auth.get_user_by_username.return_value = existing_user

    user = await guest_access.get_or_create_guest_user(mass, "party_guest", "Party Guest")

    assert user is existing_user
    mass.webserver.auth.create_user.assert_not_awaited()


async def test_get_or_create_guest_user_rejects_non_guest() -> None:
    """An existing user with a non-guest role is never reused for guest access."""
    mass = _create_mock_mass()
    mass.webserver.auth.get_user_by_username.return_value = MagicMock(role=UserRole.ADMIN)

    with pytest.raises(InvalidDataError):
        await guest_access.get_or_create_guest_user(mass, "party_guest", "Party Guest")
    mass.webserver.auth.create_user.assert_not_awaited()


async def test_get_or_create_guest_user_refuses_disabled_guest() -> None:
    """A disabled guest account is refused, never re-created or re-enabled."""
    mass = _create_mock_mass()
    mass.webserver.auth.get_user_by_username.return_value = MagicMock(
        role=UserRole.GUEST, enabled=False, display_name="Party Guest"
    )

    with pytest.raises(InvalidDataError) as exc_info:
        await guest_access.get_or_create_guest_user(mass, "party_guest", "Party Guest")

    assert exc_info.value.translation_key == "guest_account_disabled"
    assert exc_info.value.translation_args == ["Party Guest"]
    mass.webserver.auth.get_user_by_username.assert_awaited_once_with(
        "party_guest", include_disabled=True
    )
    mass.webserver.auth.create_user.assert_not_awaited()


async def test_get_or_create_guest_user_creates_guest() -> None:
    """A missing guest user is created with the GUEST role."""
    mass = _create_mock_mass()
    created_user = MagicMock()
    mass.webserver.auth.create_user.return_value = created_user

    user = await guest_access.get_or_create_guest_user(mass, "party_guest", "Party Guest")

    assert user is created_user
    mass.webserver.auth.create_user.assert_awaited_once_with(
        username="party_guest",
        role=UserRole.GUEST,
        display_name="Party Guest",
    )


async def test_get_or_create_join_code_reuses_active_code() -> None:
    """An active join code is reused instead of generating a new one."""
    mass = _create_mock_mass()
    mass.webserver.auth.get_active_join_code.return_value = "ABC123"

    code = await guest_access.get_or_create_join_code(mass, MagicMock())

    assert code == "ABC123"
    mass.webserver.auth.generate_join_code.assert_not_awaited()


async def test_get_or_create_join_code_generates_new_code() -> None:
    """A new join code is generated when no active code exists."""
    mass = _create_mock_mass()
    mass.webserver.auth.generate_join_code.return_value = ("XYZ789", MagicMock())
    user = MagicMock()

    code = await guest_access.get_or_create_join_code(
        mass, user, expires_in_hours=4, max_uses=5, device_name="Quiz Guest"
    )

    assert code == "XYZ789"
    mass.webserver.auth.generate_join_code.assert_awaited_once_with(
        user=user,
        expires_in_hours=4,
        max_uses=5,
        device_name="Quiz Guest",
    )


def test_build_join_url_remote_access() -> None:
    """The remote access URL is used when remote access is enabled."""
    mass = _create_mock_mass()
    mass.webserver.remote_access.is_enabled = True
    mass.webserver.remote_access.remote_id = "remote123"

    url = guest_access.build_join_url(mass, "ABC123")

    assert url == "https://app.music-assistant.io/?remote_id=remote123&join=ABC123"


def test_build_join_url_local() -> None:
    """The local base URL is used when remote access is disabled."""
    mass = _create_mock_mass()
    mass.webserver.remote_access.is_enabled = False
    mass.webserver.base_url = "http://192.168.1.2:8095"

    url = guest_access.build_join_url(mass, "ABC123")

    assert url == "http://192.168.1.2:8095/?join=ABC123"


def test_build_join_url_external_url_wins() -> None:
    """The external URL is preferred over both remote access and the local base URL."""
    mass = _create_mock_mass()
    mass.webserver.external_url = "https://ma.example.com/music"
    mass.webserver.remote_access.is_enabled = True
    mass.webserver.remote_access.remote_id = "remote123"
    mass.webserver.base_url = "http://192.168.1.2:8095"

    url = guest_access.build_join_url(mass, "ABC123")

    assert url == "https://ma.example.com/music/?join=ABC123"


async def test_revoke_guest_access() -> None:
    """Join codes and tokens of the guest user are revoked."""
    mass = _create_mock_mass()
    user = MagicMock()
    mass.webserver.auth.get_user_by_username.return_value = user
    mass.webserver.auth.revoke_join_codes.return_value = 2
    mass.webserver.auth.revoke_tokens_for_user.return_value = 3

    result = await guest_access.revoke_guest_access(mass, "party_guest")

    assert result == (2, 3)
    mass.webserver.auth.revoke_join_codes.assert_awaited_once_with(user)
    mass.webserver.auth.revoke_tokens_for_user.assert_awaited_once_with(user)


async def test_revoke_guest_access_disabled_guest() -> None:
    """Join codes and tokens of a disabled guest user are revoked too."""
    mass = _create_mock_mass()
    user = MagicMock(role=UserRole.GUEST, enabled=False, display_name="Party Guest")
    mass.webserver.auth.get_user_by_username.return_value = user
    mass.webserver.auth.revoke_join_codes.return_value = 1
    mass.webserver.auth.revoke_tokens_for_user.return_value = 2

    result = await guest_access.revoke_guest_access(mass, "party_guest")

    assert result == (1, 2)
    mass.webserver.auth.get_user_by_username.assert_awaited_once_with(
        "party_guest", include_disabled=True
    )
    mass.webserver.auth.revoke_join_codes.assert_awaited_once_with(user)
    mass.webserver.auth.revoke_tokens_for_user.assert_awaited_once_with(user)


async def test_revoke_guest_access_no_user() -> None:
    """Revoking access for an unknown user is a no-op."""
    mass = _create_mock_mass()

    result = await guest_access.revoke_guest_access(mass, "unknown_guest")

    assert result == (0, 0)
    mass.webserver.auth.revoke_join_codes.assert_not_awaited()


def test_credential_owner_encodes_the_lifetime_policy() -> None:
    """The owner prefix says whether a credential is session-scoped or account-bound."""
    guest = MagicMock(role=UserRole.GUEST, user_id="g1")
    user = MagicMock(role=UserRole.USER, user_id="u1")
    assert guest_access.credential_owner(guest) == "guest-g1"
    assert guest_access.credential_owner(user) == "user-u1"
    assert guest_access.is_session_scoped_owner("guest-g1")
    assert not guest_access.is_session_scoped_owner("user-u1")
    assert guest_access.credential_owners_for_user_id("x") == ("guest-x", "user-x")


def test_credential_owner_user_id_resolves_both_prefixes() -> None:
    """An owner id resolves back to its account, and other owner kinds resolve to nothing."""
    assert guest_access.credential_owner_user_id("guest-g1") == "g1"
    assert guest_access.credential_owner_user_id("user-u1") == "u1"
    assert guest_access.credential_owner_user_id("token-t1") is None


def _request(peer: str, forwarded_for: str | None = None) -> MagicMock:
    """Create a mock aiohttp request from the given peer address."""
    request = MagicMock()
    request.transport.get_extra_info.return_value = (peer, 50000)
    request.headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
    return request


def test_lan_client_address_trusts_forwarded_for_from_loopback() -> None:
    """A proxy on this host connects from loopback; the client is its last forwarded entry."""
    request = _request("127.0.0.1", "10.9.9.9, 192.168.1.20")

    assert guest_access.lan_client_address(request) == ip_address("192.168.1.20")


def test_lan_client_address_ignores_forwarded_for_from_others() -> None:
    """Only a loopback peer may name another client address."""
    request = _request("203.0.113.7", "192.168.1.20")

    assert guest_access.lan_client_address(request) == ip_address("203.0.113.7")


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("192.168.213.40", True),
        ("2604:5500:c0fc:3e00:415:d47b:c4fb:b8bb", True),
        ("fe80::1", True),
        ("127.0.0.1", True),
        ("100.114.241.7", False),
        ("192.168.1.40", False),
        ("2604:5500:aaaa::1", False),
    ],
)
def test_is_on_local_network(address: str, expected: bool) -> None:
    """Only addresses on a network of the host's own interfaces count as local."""
    adapters = [
        _adapter(("192.168.213.122", 24), (("2604:5500:c0fc:3e00::743d", 0, 0), 64)),
        _adapter((("fe80::caff:bfff:fe0b:743d", 0, 2), 64)),
        # a VPN address covers only this host, not the other peers on the VPN
        _adapter(("100.96.27.115", 32)),
    ]
    with patch.object(guest_access.ifaddr, "get_adapters", return_value=adapters):
        assert guest_access.is_on_local_network(ip_address(address)) is expected


def _adapter(*ips: tuple[str | tuple[str, int, int], int]) -> MagicMock:
    """Create a mock ifaddr adapter with the given (ip, network_prefix) pairs."""
    return MagicMock(ips=[MagicMock(ip=ip, network_prefix=prefix) for ip, prefix in ips])
