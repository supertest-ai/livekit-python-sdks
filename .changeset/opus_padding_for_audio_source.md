---
"livekit-rtc": minor-rc
---

`AudioSource.capture_frame(frame, opus_padding=...)` writes the given bytes verbatim as the Opus padding (RFC 6716, section 3.2.5) of the encoded frame holding the frame's first sample, so receivers can read them in an encoded transform while every Opus decoder decodes the frame unchanged. Raises if the source isn't published, or its room uses end-to-end encryption.
