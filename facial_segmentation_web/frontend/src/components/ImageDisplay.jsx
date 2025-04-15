import React from "react";
import { FaDownload } from "react-icons/fa";

const ImageDisplay = ({ image }) => {
  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = image;
    link.download = `facial-segmentation-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div>
      {image ? (
        <div className="space-y-3">
          <div className="relative aspect-video overflow-hidden rounded-lg bg-gray-100">
            <img
              src={image}
              alt="Segmentation Result"
              className="h-full w-full object-contain"
            />
          </div>

          <button
            onClick={handleDownload}
            className="btn btn-outline flex items-center justify-center"
          >
            <FaDownload className="mr-1" />
            Download Result
          </button>
        </div>
      ) : (
        <div className="flex aspect-video flex-col items-center justify-center rounded-lg bg-gray-100 p-6 text-center">
          <p className="mb-2 text-gray-500">No segmentation result yet</p>
          <p className="text-sm text-gray-400">
            Upload an image or take a photo and run segmentation to see the
            result here
          </p>
        </div>
      )}
    </div>
  );
};

export default ImageDisplay;
