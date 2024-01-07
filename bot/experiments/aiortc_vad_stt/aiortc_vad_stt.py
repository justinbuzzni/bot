import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import av
import torch
import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
import time
from bot.speech_recognition import SpeechRecognitionV2, SpeechRecognition
from aiortc.contrib.signaling import create_signaling

speech_recognition = SpeechRecognitionV2()


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        track,
        vad_model,
        speech_recognition: SpeechRecognition,
        peer_connection: RTCPeerConnection,
        datachannel: RTCDataChannel
        # signaling = None
    ):
        super().__init__()  # don't forget this!
        self.track = track
        self.vad_model = vad_model
        self.speech_recognition = speech_recognition
        self.peer_connection = peer_connection
        # self.signaling = signaling

        self.datachannel = datachannel

        self.sampling_rate = 16_000
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sampling_rate,
        )
        print(self.resampler)
        self.buffer = torch.tensor(
            [],
            dtype=torch.float32,
        )

        self.segments = []
        self.non_silent_segments = torch.tensor(
            [],
            dtype=torch.float32,
        )
        self.segments_amount = 80
        self.is_activated = False
        self.is_activated_threshhold = 3
        self.is_activated_amount = 0
        self.is_prev_voice = True

        self.silence_segments_offsets = 20
        self.silence_segments_amount = 0
        self.silence_segments = []

        # sst state
        self.is_sst_state = False

    async def recv(self):
        frame = await self.track.recv()
        # print(frame)
        if not self.is_sst_state:
            frame = self.resampler.resample(frame)[0]
            frame_array = frame.to_ndarray()
            frame_array = frame_array[0].astype(np.float32)
            # print(frame_array)
            # s16 (signed integer 16-bit number) can store numbers in range -32 768...32 767.
            frame_array = torch.tensor(frame_array, dtype=torch.float32) / 32_767

            speech_prob = None
            self.buffer = torch.cat(
                [
                    self.buffer,
                    frame_array,
                ]
            )
            if self.buffer.shape[0] >= frame_array.shape[0] * 4:
                speech_prob = self.vad_model(
                    self.buffer,
                    self.sampling_rate,
                ).item()

            if not speech_prob is None:
                is_speech = speech_prob >= 0.6
                # print(f"speech_prob={speech_prob}")
                if is_speech and not self.is_activated:
                    print(f"speech_prob={speech_prob}")

                    self.is_activated_amount += 1
                    self.is_prev_voice = True
                    self.silence_segments_amount = 0

                    self.non_silent_segments = torch.cat(
                        [
                            self.non_silent_segments,
                            self.buffer,
                        ]
                    )

                    if self.is_activated_amount == self.is_activated_threshhold:
                        self.is_activated = True
                        print("activated")
                        self.segments = []

                        # событие первой активации. я заметил если подавать в
                        # whisper с некоторым оффсетом, тогда он правильно распознает начало фразы
                        # тоже самое и с концом фразы
                        self.non_silent_segments = torch.cat(
                            [
                                *self.silence_segments,
                                self.non_silent_segments,
                            ]
                        )
                # после того как произошел триггер активации
                # собираем всю речь без прерываний
                elif self.is_activated:
                    self.non_silent_segments = torch.cat(
                        [
                            self.non_silent_segments,
                            self.buffer,
                        ]
                    )
                else:
                    # захват отрезка до того как классификатор речи будет очень
                    # уверен в том что я говорю
                    self.silence_segments.append(
                        self.buffer,
                    )
                    # я могу долгое время просто молчать
                    # я не хочу сохранять 10 минут тишины
                    self.silence_segments = self.silence_segments[
                        -self.silence_segments_offsets :
                    ]

                self.segments.append(int(is_speech))
                self.segments = self.segments[-self.segments_amount :]
                print(np.mean(self.segments))

                if (
                    np.mean(self.segments) <= 0.4
                    and self.is_activated
                    and len(self.segments) == self.segments_amount
                ):
                    print("Let's Speech to text!")
                    self.is_activated_amount = 0
                    self.is_activated = False
                    self.segments = []
                    self.is_sst_state = True
                    self.silence_segments = []
                    # asyncio.create_task(
                    #     asyncio.to_thread(self.extract_text),
                    # )
                    self.extract_text()

                self.buffer = frame_array

        # print(self.is_sst_state)
        return frame

    def extract_text(self):
        print("Extract text")
        # print(self.non_silent_segments)
        text = self.speech_recognition.generate(
            y=self.non_silent_segments,
            resample=False,
        )
        print(text)
        sf.write(
            "bot/experiments/aiortc_vad_stt/temp_speech.wav",
            data=self.non_silent_segments,
            samplerate=self.sampling_rate,
        )
        self.non_silent_segments = torch.tensor([], dtype=torch.float32)
        self.datachannel.send(text)
        self.is_sst_state = False


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
    stt_response_dc = pc.createDataChannel(
        label="stt_response",
        ordered=True,
    )
    # signaling = create_signaling()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("on_datachannel", channel)

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str):
                channel.send("pong" + message)

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
                vad_model=model,
                speech_recognition=speech_recognition,
                peer_connection=pc,
                datachannel=stt_response_dc,
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
