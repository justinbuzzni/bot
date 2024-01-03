const audio = document.querySelector("audio");

function handleSuccess(stream) {
  window.stream = stream; // make variable available to browser console
  audio.srcObject = stream;
  console.log(stream);
  console.log(stream.getAudioTracks()[0]);
}

function handleError(error) {
  console.log(error);
}

navigator.mediaDevices
  .getUserMedia({
    audio: true,
    video: false,
  })
  .then(handleSuccess)
  .catch(handleError);
