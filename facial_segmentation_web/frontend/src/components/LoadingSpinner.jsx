import React from "react";

const LoadingSpinner = () => {
  return (
    <div className="flex aspect-video flex-col items-center justify-center rounded-lg bg-gray-100 p-6">
      <div className="flex flex-col items-center justify-center">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600"></div>
        <p className="mt-4 text-center text-gray-600">Processing image...</p>
        <p className="mt-1 text-center text-sm text-gray-400">
          This may take a few seconds
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
