//create a new FormData object
let VideoStream;
async () => {
  VideoStream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
  document.getElementById("vidLive").srcObject = VideoStream;
  };

// Get the video element
// const video = document.getElementById('vidLive');

// // Create a canvas element
// var canvas = document.createElement('canvas');
// canvas.width = video.videoWidth;
// canvas.height = video.videoHeight;
// var ctx = canvas.getContext('2d');

// // Create an array to store the frames
// const frames = [];

// // Set the interval between frames (in milliseconds)
// const frameInterval = 1000;

// // Set the maximum number of frames to send in a batch
// const batchSize = 16;

// // Add a listener for the 'play' event on the video element
// video.addEventListener('playing', function() {
//   // Create a new function to extract and send the frames
//   const extractAndSendFrames = function() {
//     // If the video has ended, stop extracting and sending frames
//     if (video.ended) {
//       return;
//     }

//     // Draw the current frame on the canvas
//     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//     // Add the current frame to the array
//     frames.push(canvas.toDataURL());

//     // If the number of frames in the array equals the batch size, send the batch to the server
//     if (frames.length === batchSize) {
//       // Send the batch of frames to the Python backend endpoint using a POST request
//       const csrfToken = document.getElementsByName('csrfmiddlewaretoken')[0].value;
//       fetch('assign_value', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//           'X-CSRFTOKEN': csrfToken
//         },
//         body: JSON.stringify({ frames: frames })
//       })
      
//       .then(response=>response.json())
//       .then(data=> {
//         document.getElementById("Val").innerHTML = data['value'];
//       })
//       .catch(function(error) {
//         // Handle errors that may occur during the request
//         console.error(error);
//       });

//       // Clear the frames array
//       frames.splice(0, frames.length);
//     }

//     // Set a timeout to call the function again after the specified interval
//     setTimeout(extractAndSendFrames, frameInterval);
//   };

//   // Call the function to start extracting and sending frames
//   extractAndSendFrames();
// });
