---
"livekit-rtc": minor-rc
---

Add `AudioFrameEvent.opus_padding`: the Opus padding of each received packet whose first sample falls in the frame, in rooms with `RoomOptions(receive_opus_padding=True)`. Padding is now sent only on tracks published with `TrackPublishOptions(opus_padding=True)`, also with E2EE, and `capture_frame` no longer raises while no sender carries it.
