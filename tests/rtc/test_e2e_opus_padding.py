"""End-to-end test of Opus padding, from AudioSource.capture_frame to AudioFrameEvent.

The publisher sends bursts of tone, each TONE chunks of 10 ms followed by
silence, one every PERIOD chunks. The first chunk of each burst carries padding
naming that chunk. The subscriber finds each burst's onset in the decoded audio
and checks that the burst's padding arrived just before it.

Requires LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET; skipped otherwise.
"""

import asyncio
import os
import uuid
from typing import Any

import numpy as np
import pytest
from livekit import api, rtc

RATE = 48000
CHUNK = RATE // 100
PERIOD = 8
TONE = 4
CHUNKS = 50 * PERIOD
# Audio the subscriber may conceal while playout starts. Padding of concealed
# audio is not delivered.
STARTUP_CHUNKS = 7 * PERIOD
# Latest a burst's onset may follow the start of the chunk carrying its
# padding: the chunk (10 ms), the start of the packet holding the burst's first
# sample (up to 20 ms earlier), Opus's algorithmic delay (6.5 ms), and slack for
# the jitter buffer stretching audio.
MAX_ONSET_DELAY = 40 * CHUNK // 10
LOUD = 4000


def skip_if_no_credentials() -> Any:
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]
    return pytest.mark.skipif(
        bool(missing), reason=f"Missing environment variables: {', '.join(missing)}"
    )


def create_token(identity: str, room_name: str) -> str:
    grants = api.VideoGrants(room_join=True, room=room_name)
    return api.AccessToken().with_identity(identity).with_grants(grants).to_jwt()


def padding(i: int) -> bytes:
    """The padding of chunk `i`: an extension byte, then `i`."""
    return b"\xf0" + i.to_bytes(4, "big")


def chunk(i: int) -> rtc.AudioFrame:
    frame = rtc.AudioFrame.create(RATE, 1, CHUNK)
    if i < CHUNKS and i % PERIOD < TONE:
        t = (i * CHUNK + np.arange(CHUNK)) / RATE
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        np.frombuffer(frame.data, dtype=np.int16)[:] = tone
    return frame


async def send(source: rtc.AudioSource) -> None:
    """Sends the bursts, then silence until cancelled."""
    i = 0
    while True:
        if i < CHUNKS and i % PERIOD == 0:
            await source.capture_frame(chunk(i), opus_padding=padding(i))
        else:
            await source.capture_frame(chunk(i))
        i += 1


async def receive(track: rtc.Track) -> list[tuple[np.ndarray, tuple[bytes, ...]]]:
    """Receives the bursts, plus slack for subscribing and the jitter buffer."""
    chunks = []
    stream = rtc.AudioStream.from_track(track=track, sample_rate=RATE, num_channels=1)
    async for event in stream:
        samples = np.frombuffer(event.frame.data, dtype=np.int16).copy()
        assert len(samples) == CHUNK, "expected 10 ms frames"
        chunks.append((samples, tuple(event.opus_padding)))
        if len(chunks) >= CHUNKS + 200:
            break
    await stream.aclose()
    return chunks


def check(chunks: list[tuple[np.ndarray, tuple[bytes, ...]]]) -> None:
    """Checks that every padding arrived once, in order, unchanged, just before
    its burst's onset, and that only startup padding is missing."""
    # Where a burst's audio starts: a loud sample after 20 ms without one.
    samples = np.concatenate([s for s, _ in chunks]).astype(np.int32)
    loud = np.flatnonzero(np.abs(samples) > LOUD)
    onsets = loud[np.diff(loud, prepend=-(10**9)) > 2 * CHUNK]

    delivered = {}
    last = -1
    for c, (_, paddings) in enumerate(chunks):
        for p in paddings:
            assert len(p) == 5 and p[0] == 0xF0, f"padding {p.hex()} was never sent"
            i = int.from_bytes(p[1:], "big")
            assert i < CHUNKS and i % PERIOD == 0, f"padding for chunk {i} was never sent"
            assert i > last, f"padding for chunk {i} after chunk {last}"
            last = i
            delivered[i] = c * CHUNK

    for i in range(STARTUP_CHUNKS, CHUNKS, PERIOD):
        assert i in delivered, f"padding for chunk {i} missing"
        at = delivered[i]
        after = onsets[onsets >= at]
        assert len(after), "no onset after padding"
        delay = int(after[0]) - at
        assert delay < MAX_ONSET_DELAY, (
            f"padding for chunk {i} arrived {delay * 10 // CHUNK} ms before its burst"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("red", [False, True])
@skip_if_no_credentials()  # type: ignore[untyped-decorator]
async def test_opus_padding(red: bool) -> None:
    room_name = f"test-opus-padding-{uuid.uuid4().hex[:8]}"
    url = os.environ["LIVEKIT_URL"]
    publisher, subscriber = rtc.Room(), rtc.Room()
    subscribed: asyncio.Queue[rtc.Track] = asyncio.Queue()

    @subscriber.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            subscribed.put_nowait(track)

    await subscriber.connect(url, create_token("subscriber", room_name))
    await publisher.connect(url, create_token("publisher", room_name))
    try:
        source = rtc.AudioSource(RATE, 1, queue_size_ms=100)
        track = rtc.LocalAudioTrack.create_audio_track("padded", source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE, red=red, dtx=False
        )
        await publisher.local_participant.publish_track(track, options)
        sender = asyncio.create_task(send(source))
        try:
            remote = await asyncio.wait_for(subscribed.get(), 10)
            check(await asyncio.wait_for(receive(remote), 20))
        finally:
            sender.cancel()
            await source.aclose()
    finally:
        await publisher.disconnect()
        await subscriber.disconnect()
