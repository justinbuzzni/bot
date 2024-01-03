// peer connection
var pc = null;

// data channel
var dc = null,
  dcInterval = null;

function createPeerConnection() {
  pc = new RTCPeerConnection({
    sdpSemantics: "unified-plan",
  });

  pc.addEventListener("icegatheringstatechange", function () {}, false);

  pc.addEventListener("iceconnectionstatechange", function () {}, false);

  pc.addEventListener("signalingstatechange", function () {}, false);

  // connect audio / video
  pc.addEventListener("track", function (evt) {
    document.getElementById("audio").srcObject = evt.streams[0];
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
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;

      return fetch("/offer", {
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          video_transform: false,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    })
    .then(function (response) {
      return response.json();
    })
    .then(function (answer) {
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      alert(e);
    });
}

function start() {
  document.getElementById("start").style.display = "none";

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

  dc = pc.createDataChannel("chat", parameters);
  dc.onclose = function () {
    clearInterval(dcInterval);
  };

  dc.onopen = function () {
    dcInterval = setInterval(function () {
      var message = "ping " + current_stamp();
      console.log(message);
      dc.send(message);
    }, 1000);
  };
  dc.onmessage = function (evt) {
    console.log(evt);
  };

  var constraints = {
    audio: true,
    video: false,
  };

  if (constraints.audio || constraints.video) {
    navigator.mediaDevices.getUserMedia(constraints).then(
      function (stream) {
        stream.getTracks().forEach(function (track) {
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
