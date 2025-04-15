import React, { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { FaCamera, FaRedo } from "react-icons/fa";

const CameraCapture = ({ onCapture }) => {
  const webcamRef = useRef(null);
  const [isCaptured, setIsCaptured] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);

  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
      setIsCaptured(true);

      if (onCapture) {
        onCapture(imageSrc);
      }
    }
  }, [webcamRef, onCapture]);

  const reset = useCallback(() => {
    setCapturedImage(null);
    setIsCaptured(false);
  }, []);

  const videoConstraints = {
    width: 640,
    height: 360,
    facingMode: "user",
  };

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-lg bg-gray-100">
        {!isCaptured ? (
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className="h-full w-full object-cover"
          />
        ) : (
          <img
            src={capturedImage}
            alt="Captured"
            className="h-full w-full object-cover"
          />
        )}
      </div>

      <div className="flex justify-center space-x-3">
        {!isCaptured ? (
          <button
            onClick={capture}
            className="btn btn-primary flex items-center"
            type="button"
          >
            <FaCamera className="mr-1.5" />
            Take Photo
          </button>
        ) : (
          <button
            onClick={reset}
            className="btn btn-outline flex items-center"
            type="button"
          >
            <FaRedo className="mr-1.5" />
            Retake
          </button>
        )}
      </div>
    </div>
  );
};

export default CameraCapture;
