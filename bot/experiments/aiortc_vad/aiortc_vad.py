import argparse
import asyncio
import json
import logging
import os
import uuid

# Set TORCH_HOME to control model cache location
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.torch')

import aiohttp
import av
import numpy as np
import torch
from aiohttp import web
from aiohttp_cors import CorsViewMixin, ResourceOptions
from aiohttp_cors import setup as cors_setup
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
import time

import librosa
import torchaudio

# Preload the model to avoid loading it for each connection
try:
    print("Preloading Silero VAD model...")
    silero_model, silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=True,
    )
    print("Silero VAD model loaded successfully")
except Exception as e:
    print(f"Error preloading Silero VAD model: {e}")
    silero_model = None
    silero_utils = None
    import traceback
    traceback.print_exc()

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
        
        if utils is None:
            # If model loading failed, create empty placeholders
            self.get_speech_timestamps = lambda *args, **kwargs: []
            self.save_audio = lambda *args, **kwargs: None
            self.read_audio = lambda *args, **kwargs: None
            self.VADIterator = lambda *args, **kwargs: None
            self.collect_chunks = lambda *args, **kwargs: None
        else:
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

        self.sampling_rate = 16_000  # VAD model requires 16kHz audio
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
        
        # For debug info
        self.frame_count = 0
        print(f"AudioTrack initialized with sampling_rate={self.sampling_rate}")

    async def recv(self):
        try:
            frame = await self.track.recv()
            
            # If the VAD model is not available, just pass through the audio
            if self.vad_model is None:
                return frame
            
            # ---- RESAMPLING IS WORKING ----
            frame_resampled = self.resampler.resample(frame)[0]
            frame_array = frame_resampled.to_ndarray()[0]
            # s16 (signed integer 16-bit number) can store numbers in range -32 768...32 767.
            frame_array = torch.tensor(frame_array, dtype=torch.float32) / 32_767
            # ---- RESAMPLING IS WORKING ----
            
            # Debug info - log every 20 frames to avoid flooding
            self.frame_count += 1
            if self.frame_count % 20 == 0:
                print(f"Frame {self.frame_count}: samples={frame_array.shape[0]}, buffer before={self.buffer.shape[0]}")
            
            speech_prob = None
            self.buffer = torch.cat([self.buffer, frame_array])
            
            # Model expects exactly 512 samples for 16kHz audio
            if self.buffer.shape[0] >= 512:
                try:
                    # Use only the first 512 samples and discard the rest or keep for next frame
                    vad_input = self.buffer[:512]
                    remaining = self.buffer[512:] if self.buffer.shape[0] > 512 else torch.tensor([], dtype=torch.float32)
                    
                    if self.frame_count % 20 == 0:
                        print(f"VAD input: shape={vad_input.shape}, remaining={remaining.shape[0]}")
                    
                    speech_prob = self.vad_model(
                        vad_input,
                        self.sampling_rate,
                    ).item()
                    
                    # Keep remaining samples for next inference
                    self.buffer = remaining
                    
                except Exception as e:
                    print(f"Error in VAD model: {e}")
                    print(f"Buffer shape when error occurred: {self.buffer.shape}")
                    # Keep the most recent samples (up to 512) to avoid accumulating too much data
                    if self.buffer.shape[0] > 512:
                        self.buffer = self.buffer[-512:]
                    speech_prob = None

            if speech_prob is not None:
                is_speech = speech_prob >= 0.4
                print(f"speech_prob={speech_prob}")
                if is_speech:
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
            
        except Exception as e:
            print(f"Error in recv: {e}")
            import traceback
            traceback.print_exc()
            # Return the original frame if we can, otherwise create a silent frame
            if 'frame' in locals():
                return frame
            
            # Create a silent audio frame as fallback
            return self.track.recv()


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    try:
        # Set a higher payload size limit 
        params = await request.json(loads=json.loads)
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
            log_info("Track %s received", track)

            try:
                if track.kind == "audio":
                    audio_track = AudioTrack(
                        relay.subscribe(
                            track=track,
                        ),
                        model=silero_model,
                        utils=silero_utils,
                    )
                    recorder.addTrack(audio_track)
            except Exception as e:
                log_info("Error in on_track: %s", str(e))
                import traceback
                traceback.print_exc()
                raise e

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
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Error in offer handler: %s", str(e))
        return web.Response(status=500, text=str(e))


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

    # Configure larger buffer sizes to handle SDP data
    app = web.Application(client_max_size=1024*1024*10)
    app.on_shutdown.append(on_shutdown)
    
    # Setup routes
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    
    # Setup CORS
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
            expose_headers="*",
            max_age=3600,
        )
    })
    
    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Set max header size
    aiohttp.web_runner.DEFAULT_HTTP_MAX_HEADER_SIZE = 20000

    web.run_app(
        app,
        access_log=None,
        host=args.host,
        port=args.port,
        ssl_context=ssl_context,
    )
