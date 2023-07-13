// Variables for video feed and object details
const videoElement = document.getElementById("video-feed");
const detailsContent = document.getElementById("details-content");
const start_btn = document.getElementById("start-btn");
const stop_btn = document.getElementById("stop-btn");
const Results_heading = document.getElementById("Results-heading");

// Load the pre-trained model from Teachable Machine
async function loadModel() {
  const modelURL =
    "https://storage.googleapis.com/tm-model/-E8kn0Vxi/model.json";
  const metadataURL =
    "https://storage.googleapis.com/tm-model/-E8kn0Vxi/metadata.json";
  const model = await tf.loadLayersModel(modelURL);
  const metadata = await fetch(metadataURL);
  const metadataJSON = await metadata.json();
  return { model, metadata: metadataJSON };
}

// Function to classify objects
async function classifyObject() {
  const { model, metadata } = await loadModel();

  // Capture frame from the video feed
  const image = tf.browser.fromPixels(videoElement);
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  const normalizedImage = resizedImage.toFloat().div(tf.scalar(255));
  const batchedImage = normalizedImage.expandDims(0);

  // Make predictions
  const predictions = await model.predict(batchedImage).data();
  const topPrediction = Array.from(predictions)
    .map((p, i) => {
      return {
        probability: p,
        className: metadata.labels[i],
      };
    })
    .sort((a, b) => b.probability - a.probability)[0];

  // Display object details
  if (topPrediction.probability > 0.5) {
    const className = topPrediction.className;
    detailsContent.innerHTML = `<h3 id="classname">${className}</h3>`;
    if (className == "Dim Light") {
      const classname = document.getElementById("classname");
      classname.style.color = "blue";
      detailsContent.innerHTML += "<p>🔅 Try to improve your light</p>";
    } else if (className == "Full Light") {
      const classname = document.getElementById("classname");
      classname.style.color = "green";
      detailsContent.innerHTML +=
        "<p>🌞 Great! Your lighting setup is perfect.</p>";
    } else if (className == "Very Low Light") {
      const classname = document.getElementById("classname");
      classname.style.color = "gray";
      detailsContent.innerHTML +=
        "<p>🔆 Be careful! your light is very low. Try moving near the light source.</p>";
    } else {
      const classname = document.getElementById("classname");
      classname.style.color = "red";
      detailsContent.innerHTML +=
        "<p>🌑 You are in complete dark. Try turning on the lights or moving closer to the light source.</p>";
    }
  } else {
    detailsContent.innerHTML = "<h3>No object detected</h3>";
  }

  // Clean up
  image.dispose();
  resizedImage.dispose();
  normalizedImage.dispose();
  batchedImage.dispose();
}

// Access the webcam stream
async function accessWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    videoElement.addEventListener("loadeddata", () => {
      classifyObject();
    });
  } catch (error) {
    console.error("Error accessing webcam: ", error);
  }
}

// Start the webcam stream
start_btn.addEventListener("click", () => {
  Results_heading.style.display = "block";
  detailsContent.innerHTML = null;
  accessWebcam();
  start_btn.innerText = "Check Again";
});

// Stop the webcam stream
stop_btn.addEventListener("click", () => {
  start_btn.innerText = "Check Your Light";
  Results_heading.style.display = "none";
  videoElement.srcObject.getTracks().forEach((track) => track.stop());
  detailsContent.innerHTML =
    "<p>Check your lighting setup by just pressing the <span style='font-weight: bolder; color:green'>Check Your Light</span> button 🙂</p><img src='https://png.pngtree.com/element_our/20190529/ourmid/pngtree-stereo-start-button-illustration-image_1215553.jpg' alt='placeholder'>";
});
