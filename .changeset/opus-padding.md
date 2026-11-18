---
"livekit-rtc": minor
---

Carry caller-supplied bytes in the Opus padding of audio frames: `AudioSource.capture_frame(frame, opus_padding=...)` writes them into the encoded frame holding the frame's first sample, and `AudioFrameEvent.opus_padding` returns the padding of each received packet whose first sample falls in the frame. Opt-in on each side: `TrackPublishOptions(opus_padding=True)` and `RoomOptions(receive_opus_padding=True)`, also with E2EE.
