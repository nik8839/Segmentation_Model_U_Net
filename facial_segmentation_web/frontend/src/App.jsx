import { useState, useEffect } from "react";
import Header from "./components/Header";
import SegmentationForm from "./components/SegmentationForm";
import ImageDisplay from "./components/ImageDisplay";
import CameraCapture from "./components/CameraCapture";
import LoadingSpinner from "./components/LoadingSpinner";
import { toast } from "react-toastify";
import axios from "axios";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState([
    "eye",
    "nose",
    "lips",
  ]);
  const [outputType, setOutputType] = useState("overlay");
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [apiStatus, setApiStatus] = useState({ connected: false, models: {} });

  // Check API status on mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get("/api/status");
      setApiStatus({
        connected: true,
        models: response.data.models || {},
        device: response.data.device || "unknown",
      });
    } catch (error) {
      setApiStatus({ connected: false, models: {} });
      toast.error(
        "Cannot connect to segmentation API. Please ensure the backend server is running."
      );
    }
  };

  const handleImageUpload = (imageFile) => {
    setIsCameraActive(false);
    setSegmentedImage(null);

    if (!imageFile) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target.result);
    };
    reader.readAsDataURL(imageFile);
  };

  const handleCameraCapture = (imageSrc) => {
    setSelectedImage(imageSrc);
    setSegmentedImage(null);
    setIsCameraActive(false);
  };

  const handleSegmentation = async () => {
    if (!selectedImage) {
      toast.warning("Please select or capture an image first");
      return;
    }

    if (!apiStatus.connected) {
      toast.error(
        "Cannot connect to segmentation API. Please ensure the backend server is running."
      );
      return;
    }

    if (selectedFeatures.length === 0) {
      toast.warning("Please select at least one facial feature to segment");
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post("/api/segment", {
        image: selectedImage,
        features: selectedFeatures,
        outputType: outputType,
      });

      setSegmentedImage(response.data.segmented_image);
      toast.success("Segmentation completed successfully");
    } catch (error) {
      console.error("Segmentation error:", error);
      toast.error("Failed to segment image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCamera = () => {
    setIsCameraActive(!isCameraActive);
    if (!isCameraActive) {
      setSegmentedImage(null);
    }
  };

  return (
    <div className="min-h-screen">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          <div className="space-y-6">
            <div className="card">
              <h2 className="mb-4 text-xl font-semibold">Input Options</h2>

              <div className="mb-4 flex space-x-3">
                <button
                  className={`btn ${
                    isCameraActive ? "btn-primary" : "btn-outline"
                  }`}
                  onClick={toggleCamera}
                >
                  {isCameraActive ? "Camera Active" : "Use Camera"}
                </button>

                {!isCameraActive && (
                  <label className="btn btn-outline cursor-pointer">
                    <span>Upload Image</span>
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleImageUpload(e.target.files[0])}
                    />
                  </label>
                )}
              </div>

              {isCameraActive ? (
                <CameraCapture onCapture={handleCameraCapture} />
              ) : (
                <div className="relative aspect-video overflow-hidden rounded-lg bg-gray-100">
                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <p className="text-gray-500">No image selected</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            <SegmentationForm
              selectedFeatures={selectedFeatures}
              setSelectedFeatures={setSelectedFeatures}
              outputType={outputType}
              setOutputType={setOutputType}
              onSegment={handleSegmentation}
              isLoading={isLoading}
              apiStatus={apiStatus}
            />
          </div>

          <div className="card">
            <h2 className="mb-4 text-xl font-semibold">Segmentation Result</h2>

            {isLoading ? (
              <LoadingSpinner />
            ) : (
              <ImageDisplay image={segmentedImage} />
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-200 py-4 text-center text-sm text-gray-500">
        <p>Facial Feature Segmentation App &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
