// peer connection
var pc = null;

// data channel
var dc = null,
  dcInterval = null;

function createPeerConnection() {
  var config = {
    iceServers: [
      {urls: ['stun:stun.l.google.com:19302']}
    ],
    // Limit ICE candidates to reduce SDP size
    iceTransportPolicy: 'all',
    iceCandidatePoolSize: 2
  };
  
  pc = new RTCPeerConnection(config);
  console.log("new RTCPeerConnection", pc);

  pc.addEventListener(
    "icegatheringstatechange",
    function (event) {
      console.log("icegatheringstatechange", event);
    },
    false
  );

  pc.addEventListener(
    "iceconnectionstatechange",
    function (event) {
      console.log("iceconnectionstatechange", event);
    },
    false
  );

  pc.addEventListener(
    "signalingstatechange",
    function (event) {
      console.log("signalingstatechange", event);
    },
    false
  );

  // connect audio / video
  // когда происходит событие track, мы получаем объект
  /**
   RTCTrackEvent {isTrusted: true, receiver: RTCRtpReceiver, track: MediaStreamTrack, streams: Array(1), transceiver: RTCRtpTransceiver, …}
      isTrusted: true
      bubbles: false
      cancelBubble: false
      cancelable: false
      composed: false
      currentTarget: RTCPeerConnection {localDescription: RTCSessionDescription, currentLocalDescription: RTCSessionDescription, pendingLocalDescription: null, remoteDescription: RTCSessionDescription, currentRemoteDescription: RTCSessionDescription, …}
      defaultPrevented: false
      eventPhase: 0
      receiver: RTCRtpReceiver {track: MediaStreamTrack, transport: RTCDtlsTransport, rtcpTransport: null, playoutDelayHint: null}
      returnValue: true
      srcElement: RTCPeerConnection {localDescription: RTCSessionDescription, currentLocalDescription: RTCSessionDescription, pendingLocalDescription: null, remoteDescription: RTCSessionDescription, currentRemoteDescription: RTCSessionDescription, …}
      streams:Array(1)
        0: ==== ВОТ ТУТ НАШ СТРИМ ==== 
          MediaStream {id: '74d9539e-72d0-4c6f-910c-c1aa69945245', active: false, onaddtrack: null, onremovetrack: null, onactive: null, …}
        length: 1
      target: RTCPeerConnection {localDescription: RTCSessionDescription, currentLocalDescription: RTCSessionDescription, pendingLocalDescription: null, remoteDescription: RTCSessionDescription, currentRemoteDescription: RTCSessionDescription, …}
      timeStamp: 2296.899999976158
      track: MediaStreamTrack {kind: 'audio', id: '87781a53-b325-413b-ac55-0dfa9f631b68', label: '87781a53-b325-413b-ac55-0dfa9f631b68', enabled: true, muted: false, …}
      transceiver: RTCRtpTransceiver {mid: null, sender: RTCRtpSender, receiver: RTCRtpReceiver, stopped: true, direction: 'stopped', …}
      type: "track"
      RTCTrackEvent
   */
  // в этом объекте содержится стрим, который мы можем назначить в audio
  // и аудио будет воспроизводить данный поток
  pc.addEventListener("track", function (evt) {
    document.getElementById("audio").srcObject = evt.streams[0];
    console.log("pc.addEventListener(track", evt);
  });

  return pc;
}

function negotiate() {
  return pc
    .createOffer()
    .then(function (offer) {
      return pc.setLocalDescription(offer);
    })
    .then(function () {
      // wait for ICE gathering to complete
      return new Promise(function (resolve) {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          function checkState() {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          }
          pc.addEventListener("icegatheringstatechange", checkState);
          
          // Set a timeout to resolve anyway after 2 seconds
          // This helps prevent waiting indefinitely for ICE gathering
          setTimeout(function() {
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }, 2000);
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;
      console.log("SDP size:", JSON.stringify(offer).length, "bytes");

      // отправка SDP
      return fetch("http://localhost:8080/offer", {
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          video_transform: false,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
        credentials: "include",
        mode: "cors"
      });
    })
    .then(function (response) {
      if (!response.ok) {
        throw new Error('HTTP error, status = ' + response.status);
      }
      return response.json();
    })
    .then(function (answer) {
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      console.error("Negotiation error:", e);
      alert("Connection error: " + e);
    });
}

function start() {
  document.getElementById("start").style.display = "none";
  // создаем соединение
  pc = createPeerConnection();

  var time_start = null;

  function current_stamp() {
    if (time_start === null) {
      time_start = new Date().getTime();
      return 0;
    } else {
      return new Date().getTime() - time_start;
    }
  }

  var parameters = { ordered: true };

  // создает data channel
  dc = pc.createDataChannel("chat", parameters);
  dc.onclose = function () {
    console.log("dc.onclose");
    clearInterval(dcInterval);
  };

  dc.onopen = function () {
    dcInterval = setInterval(function () {
      var message = "ping " + current_stamp();
      console.log(message);
      dc.send("dc.onopen", message);
    }, 1000);
  };
  dc.onmessage = function (evt) {
    console.log("dc.onmessage", evt);
  };

  var constraints = {
    audio: true,
    video: false,
  };

  if (constraints.audio || constraints.video) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const errorMsg = 'Browser API navigator.mediaDevices.getUserMedia not available';
      alert(errorMsg);
      console.error(errorMsg);
      document.getElementById("stop").style.display = "inline-block";
      return negotiate();
    }
    
    navigator.mediaDevices.getUserMedia(constraints).then(
      function (stream) {
        stream.getTracks().forEach(function (track) {
          console.log(track);
          pc.addTrack(track, stream);
        });
        return negotiate();
      },
      function (err) {
        alert("Could not acquire media: " + err);
      }
    );
  } else {
    negotiate();
  }

  document.getElementById("stop").style.display = "inline-block";
}

function stop() {
  document.getElementById("stop").style.display = "none";

  // close data channel
  if (dc) {
    dc.close();
  }

  // close transceivers
  if (pc.getTransceivers) {
    pc.getTransceivers().forEach(function (transceiver) {
      if (transceiver.stop) {
        transceiver.stop();
      }
    });
  }

  // close local audio / video
  pc.getSenders().forEach(function (sender) {
    sender.track.stop();
  });

  // close peer connection
  setTimeout(function () {
    pc.close();
  }, 500);
}
