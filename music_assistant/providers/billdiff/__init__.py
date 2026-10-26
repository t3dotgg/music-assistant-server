"""
billdiff music provider for Music Assistant.

Plays the billdifferen playlist archive (https://billdifferen.blogspot.com). The playlists come
from the pilldiff catalog (https://pilldiff.t3.gg), which mixes YouTube and SoundCloud tracks in
the blog's order. yt-dlp finds each track's audio on the server, so nothing needs a login.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import socket
import time
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import aiohttp
from music_assistant_models.enums import (
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    MediaNotFoundError,
    ProviderUnavailableError,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemMetadata,
    MediaItemType,
    Playlist,
    ProviderMapping,
    Track,
)
from music_assistant_models.streamdetails import StreamDetails
from music_assistant_models.unique_list import UniqueList

from music_assistant.helpers.util import install_package
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {ProviderFeature.BROWSE}

CATALOG_URL = "https://pilldiff.t3.gg/catalog.json"
CATALOG_MAX_AGE = 6 * 3600
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)
# yt-dlp changes often to keep up with YouTube, so it is installed at runtime, not pinned
YT_DLP_PACKAGE = "yt-dlp[default]"
DEFAULT_STREAM_URL_EXPIRATION = 600


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return BilldiffProvider(mass, manifest, config, SUPPORTED_FEATURES)


class BilldiffProvider(MusicProvider):
    """The billdifferen playlists, played from YouTube and SoundCloud."""

    _catalog_fetched: float = 0
    _playlists: dict[str, dict[str, Any]]
    # track id -> (catalog entry, the playlist it is in)
    _entries: dict[str, tuple[dict[str, Any], dict[str, Any]]]
    _catalog_lock: asyncio.Lock
    _ipv4_session: aiohttp.ClientSession

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to setup this provider."""
        return ()

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return True

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._playlists = {}
        self._entries = {}
        self._catalog_lock = asyncio.Lock()
        self._ipv4_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(family=socket.AF_INET)
        )
        logging.getLogger("yt_dlp").setLevel(self.logger.level + 10)
        # a failed install must not stop the server, so playback reports it instead
        try:
            await install_package(YT_DLP_PACKAGE)
        except RuntimeError as err:
            self.logger.warning("Could not install yt-dlp: %s", err)

    async def browse(self, path: str) -> Sequence[MediaItemType | BrowseFolder]:
        """Browse the playlists, newest first."""
        await self._load_catalog()
        return [self._parse_playlist(p) for p in self._playlists.values()]

    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id."""
        await self._load_catalog()
        if not (playlist := self._playlists.get(prov_playlist_id)):
            msg = f"billdiff playlist {prov_playlist_id} not found"
            raise MediaNotFoundError(msg)
        return self._parse_playlist(playlist)

    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Get all playlist tracks for given playlist id."""
        if page > 0:
            return []
        await self._load_catalog()
        if not (playlist := self._playlists.get(prov_playlist_id)):
            msg = f"billdiff playlist {prov_playlist_id} not found"
            raise MediaNotFoundError(msg)
        return [
            self._parse_track(entry, playlist)
            for entry in playlist["tracks"]
            if entry["id"] in self._entries
        ]

    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        entry, playlist = await self._get_entry(prov_track_id)
        return self._parse_track(entry, playlist)

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get artist details by id. The catalog has only artist names, so the id is the name."""
        return Artist(
            item_id=prov_artist_id,
            provider=self.instance_id,
            name=prov_artist_id,
            provider_mappings={
                ProviderMapping(
                    item_id=prov_artist_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given track when it will be streamed."""
        entry, _ = await self._get_entry(item_id)
        info = await asyncio.to_thread(self._extract_audio, entry["playbackUrl"])
        url: str = info["url"]
        expiration = DEFAULT_STREAM_URL_EXPIRATION
        query = parse_qs(urlparse(url).query)
        for param in ("expire", "Expires", "expires"):
            if expire_ts := query.get(param, [None])[0]:
                expiration = max(30, int(expire_ts) - int(time.time()) - 10)
                break
        duration = int(info["duration"]) if info.get("duration") else None
        if entry["provider"] == "youtube":
            # A YouTube URL only works from the IPv4 address that yt-dlp used. ffmpeg would
            # connect over IPv6 when the CDN has an IPv6 address and get a 403, so we fetch
            # the audio ourselves (get_audio_stream) and MA seeks by skipping decoded audio.
            return StreamDetails(
                provider=self.instance_id,
                item_id=item_id,
                # let ffmpeg work out the format (usually Opus in WebM)
                audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
                stream_type=StreamType.CUSTOM,
                data={"url": url, "headers": info.get("http_headers") or {}},
                duration=duration,
                can_seek=False,
                allow_seek=True,
                expiration=expiration,
                # YouTube throttles delivery to about playback speed
                is_realtime=True,
            )
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            # let ffmpeg work out the format, SoundCloud uses several
            audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
            stream_type=StreamType.HLS
            if str(info.get("protocol", "")).startswith("m3u8")
            else StreamType.HTTP,
            path=url,
            duration=duration,
            can_seek=True,
            allow_seek=True,
            expiration=expiration,
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Return the audio of a YouTube track, fetched over IPv4."""
        async with self._ipv4_session.get(
            streamdetails.data["url"],
            headers=streamdetails.data["headers"],
            timeout=aiohttp.ClientTimeout(sock_connect=15, sock_read=60),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                yield chunk

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        await self._ipv4_session.close()
        await super().unload(is_removed)

    def _extract_audio(self, url: str) -> dict[str, Any]:
        """
        Find the best audio stream for a YouTube or SoundCloud page with yt-dlp.

        :param url: The YouTube or SoundCloud URL from the catalog.
        :raises UnplayableMediaError: yt-dlp is missing or can not play this URL.
        """
        try:
            yt_dlp = importlib.import_module("yt_dlp")
        except ImportError as err:
            raise UnplayableMediaError("yt-dlp is not installed") from err
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "bestaudio/best",
            # IPv4 only: a host that has IPv6 addresses but no IPv6 route makes yt-dlp hang
            "source_address": "0.0.0.0",  # noqa: S104 - picks IPv4, binds no server
            "socket_timeout": 15,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info: dict[str, Any] = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as err:
                raise UnplayableMediaError(str(err)) from err
        if not info.get("url"):
            msg = f"No audio stream found for {url}"
            raise UnplayableMediaError(msg)
        return info

    async def _get_entry(self, track_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the catalog entry for a track id and the playlist it is in."""
        await self._load_catalog()
        if not (found := self._entries.get(track_id)):
            msg = f"billdiff track {track_id} not found"
            raise MediaNotFoundError(msg)
        return found

    async def _load_catalog(self) -> None:
        """Fetch the catalog when it is missing or older than CATALOG_MAX_AGE."""
        async with self._catalog_lock:
            if self._playlists and time.time() - self._catalog_fetched < CATALOG_MAX_AGE:
                return
            try:
                async with self.mass.http_session.get(CATALOG_URL, timeout=HTTP_TIMEOUT) as resp:
                    resp.raise_for_status()
                    catalog: dict[str, Any] = await resp.json()
            except (aiohttp.ClientError, TimeoutError, ValueError) as err:
                if self._playlists:
                    # keep playing from the old catalog until the next fetch works
                    self.logger.warning("Could not refresh the billdiff catalog: %s", err)
                    return
                msg = f"billdiff catalog unavailable: {err}"
                raise ProviderUnavailableError(msg) from err
            self._playlists = {p["id"]: p for p in catalog["playlists"]}
            # an entry of kind "playlist" is a whole album or set, which we skip
            self._entries = {
                entry["id"]: (entry, playlist)
                for playlist in catalog["playlists"]
                for entry in playlist["tracks"]
                if entry["kind"] == "track"
            }
            self._catalog_fetched = time.time()

    def _parse_playlist(self, playlist: dict[str, Any]) -> Playlist:
        """Convert a catalog playlist to a Music Assistant Playlist."""
        result = Playlist(
            item_id=playlist["id"],
            provider=self.instance_id,
            name=playlist["shortTitle"],
            owner="billdifferen",
            is_editable=False,
            metadata=MediaItemMetadata(description=playlist["title"]),
            provider_mappings={
                ProviderMapping(
                    item_id=playlist["id"],
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=playlist["sourceUrl"],
                )
            },
        )
        if artwork := playlist.get("artworkUrl"):
            result.metadata.add_image(self._image(artwork))
        return result

    def _parse_track(self, entry: dict[str, Any], playlist: dict[str, Any]) -> Track:
        """Convert a catalog entry to a Music Assistant Track."""
        artist = entry["artist"] or "Unknown artist"
        track = Track(
            item_id=entry["id"],
            provider=self.instance_id,
            name=entry["title"],
            position=entry["position"],
            artists=UniqueList(
                [
                    ItemMapping(
                        media_type=MediaType.ARTIST,
                        item_id=artist,
                        provider=self.instance_id,
                        name=artist,
                    )
                ]
            ),
            metadata=MediaItemMetadata(description=entry.get("description")),
            provider_mappings={
                ProviderMapping(
                    item_id=entry["id"],
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=entry["sourceUrl"],
                )
            },
        )
        # YouTube has a thumbnail for every video; SoundCloud tracks get the post's artwork
        if video_id := entry.get("videoId"):
            track.metadata.add_image(
                self._image(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
            )
        elif artwork := playlist.get("artworkUrl"):
            track.metadata.add_image(self._image(artwork))
        return track

    def _image(self, url: str) -> MediaItemImage:
        return MediaItemImage(
            type=ImageType.THUMB, path=url, provider=self.instance_id, remotely_accessible=True
        )
