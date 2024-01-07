import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import av
import torch
import librosa
import numpy as np

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
import time
import librosa
import torchaudio


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        track,
        model,
        utils,
    ):
        super().__init__()  # don't forget this!
        self.track = track
        self.vad_model = model
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = utils
        self.get_speech_timestamps = get_speech_timestamps
        self.save_audio = save_audio
        self.read_audio = read_audio
        self.VADIterator = VADIterator
        self.collect_chunks = collect_chunks

        self.sampling_rate = 16_000
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sampling_rate,
        )
        self.torch_resampler = torchaudio.transforms.Resample(
            48_000,
            16_000,
        )
        self.buffer = torch.tensor([], dtype=torch.float32)

        self.segments = []
        self.segments_amount = 80
        self.is_activated = False
        self.is_activated_threshhold = 5
        self.is_activated_amount = 0

    async def recv(self):
        frame = await self.track.recv()
        # print(frame.to_ndarray()[0].shape, frame.rate)
        # ---- RESAMPLING NOT WORKING FROM 3rd party packages ----
        # frame_array = frame.to_ndarray().astype(np.float32)
        # frame_array = frame_array[0]
        # # frame_array = (frame_array[:960] + frame_array[960:]) / 2
        # frame_array = librosa.resample(
        #     y=frame_array,
        #     orig_sr=frame.rate,
        #     target_sr=self.sampling_rate,
        # )
        # print(frame_array)
        # frame_array = (
        #     self.torch_resampler(
        #         torch.tensor(
        #             frame.to_ndarray()[0].astype(np.float32),
        #             # frame.to_ndarray()[0][:960].astype(np.float32),
        #             dtype=torch.float32,
        #         )
        #     )
        #     .squeeze()
        #     .numpy()
        # )
        # ---- RESAMPLING NOT WORKING ----

        # ---- RESAMPLING IS WORKING ----
        frame = self.resampler.resample(frame)[0]
        frame_array = frame.to_ndarray()[0]
        # s16 (signed integer 16-bit number) can store numbers in range -32 768...32 767.
        frame_array = torch.tensor(frame_array, dtype=torch.float32) / 32_767
        # frame_array = torch.tensor(frame_array, dtype=torch.float32)
        # ---- RESAMPLING IS WORKING ----
        # print(frame_array.shape)
        speech_prob = None
        self.buffer = torch.cat(
            [
                self.buffer,
                frame_array,
            ]
        )
        # if self.buffer.shape[0] < 304 * 2:
        if self.buffer.shape[0] >= frame_array.shape[0] * 4:
            speech_prob = self.vad_model(
                self.buffer,
                self.sampling_rate,
            ).item()

            self.buffer = torch.tensor(
                [],
                dtype=torch.float32,
            )

        if not speech_prob is None:
            is_speech = speech_prob >= 0.4
            print(f"speech_prob={speech_prob}")
            if is_speech:
                # print(f"speech_prob={speech_prob}")
                self.is_activated_amount += 1
                if (
                    self.is_activated_amount >= self.is_activated_threshhold
                    and not self.is_activated
                ):
                    self.is_activated = True
                    print("activated")
                    self.segments = []

            self.segments.append(int(is_speech))
            self.segments = self.segments[-self.segments_amount :]

            if (
                np.mean(self.segments) <= 0.4
                and self.is_activated
                and len(self.segments) == self.segments_amount
            ):
                print("Let's Speech to text!")
                self.is_activated_amount = 0
                self.is_activated = False
                self.segments = []

        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"],
    )

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            audio_track = AudioTrack(
                relay.subscribe(
                    track=track,
                ),
                model=model,
                utils=utils,
            )
            recorder.addTrack(audio_track)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )

    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app,
        access_log=None,
        host=args.host,
        port=args.port,
        ssl_context=ssl_context,
    )
