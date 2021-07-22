import React, { useRef } from "react";
import "./App.css";
import * as ml5 from "ml5";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;

    drawKeypoints(pose["keypoints"], 0.6, ctx);
    drawSkeleton(pose["keypoints"], 0.7, ctx);
  };

  const runPosenet = async () => {
    setInterval(() => {
      detect();
    }, 10);
  };

  const detect = async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      let pose;
      let skeleton;
      let poseLabel;

      const classifyPose = () => {
        if (pose) {
          let inputs = [];
          for (let i = 0; i < pose.keypoints.length; i++) {
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            inputs.push(x);
            inputs.push(y);
          }
          brain.classify(inputs, gotResult);
          drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
        }
      };

      const gotResult = (error, results) => {
        if (results[0].confidence > 0.75) {
          poseLabel = results[0].label.toUpperCase();
        }
        console.log("got results", poseLabel);
        classifyPose();
      };

      const posenet = ml5.poseNet(video, () => {
        // console.log("model ready");
      });

      posenet.on("pose", (poses) => {
        // console.log("poses", poses);
        if (poses.length > 0) {
          pose = poses[0].pose;
          skeleton = poses[0].skeleton;
        }
      });

      let options = {
        inputs: 34,
        outputs: 6,
        task: "classification",
        debug: true,
      };

      const brain = ml5.neuralNetwork(options);
      const modelInfo = {
        model: "model2/model.json",
        metadata: "model2/model_meta.json",
        weights: "model2/model.weights.bin",
      };

      brain.load(modelInfo, () => {
        // console.log("brain loaded");
        classifyPose();
      });
    }
  };

  runPosenet();

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
