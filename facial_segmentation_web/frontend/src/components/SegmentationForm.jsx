import { FaEye, FaSmile, FaImage } from "react-icons/fa";
import { MdFace } from "react-icons/md";
import { HiOutlineStatusOnline, HiOutlineStatusOffline } from "react-icons/hi";

const SegmentationForm = ({
  selectedFeatures,
  setSelectedFeatures,
  outputType,
  setOutputType,
  onSegment,
  isLoading,
  apiStatus,
}) => {
  // Custom function to toggle a feature in the array
  const toggleFeature = (feature) => {
    if (selectedFeatures.includes(feature)) {
      setSelectedFeatures(selectedFeatures.filter((f) => f !== feature));
    } else {
      setSelectedFeatures([...selectedFeatures, feature]);
    }
  };

  return (
    <div className="card">
      <h2 className="mb-4 text-xl font-semibold">Segmentation Options</h2>

      <div className="mb-4">
        <div className="mb-1 flex items-center text-sm font-medium text-gray-700">
          {apiStatus.connected ? (
            <div className="flex items-center text-green-500">
              <HiOutlineStatusOnline className="mr-1" />
              API Connected
            </div>
          ) : (
            <div className="flex items-center text-red-500">
              <HiOutlineStatusOffline className="mr-1" />
              API Disconnected
            </div>
          )}

          {apiStatus.connected && (
            <span className="ml-2 text-xs text-gray-500">
              ({apiStatus.device || "unknown"})
            </span>
          )}
        </div>
      </div>

      <div className="mb-6">
        <label className="mb-2 block text-sm font-medium text-gray-700">
          Select Features to Segment
        </label>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => toggleFeature("eye")}
            className={`flex items-center rounded-full px-4 py-1.5 text-sm focus:outline-none ${
              selectedFeatures.includes("eye")
                ? "bg-blue-100 text-blue-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <FaEye className="mr-1.5" />
            Eyes
          </button>

          <button
            type="button"
            onClick={() => toggleFeature("nose")}
            className={`flex items-center rounded-full px-4 py-1.5 text-sm focus:outline-none ${
              selectedFeatures.includes("nose")
                ? "bg-orange-100 text-orange-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <MdFace className="mr-1.5" />
            Nose
          </button>

          <button
            type="button"
            onClick={() => toggleFeature("lips")}
            className={`flex items-center rounded-full px-4 py-1.5 text-sm focus:outline-none ${
              selectedFeatures.includes("lips")
                ? "bg-purple-100 text-purple-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <FaSmile className="mr-1.5" />
            Lips
          </button>
        </div>
      </div>

      <div className="mb-6">
        <label className="mb-2 block text-sm font-medium text-gray-700">
          Output Type
        </label>

        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => setOutputType("overlay")}
            className={`flex items-center rounded-full px-4 py-1.5 text-sm focus:outline-none ${
              outputType === "overlay"
                ? "bg-primary-100 text-primary-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <FaImage className="mr-1.5" />
            Color Overlay
          </button>

          <button
            type="button"
            onClick={() => setOutputType("mask")}
            className={`flex items-center rounded-full px-4 py-1.5 text-sm focus:outline-none ${
              outputType === "mask"
                ? "bg-primary-100 text-primary-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <MdFace className="mr-1.5" />
            Mask Only
          </button>
        </div>
      </div>

      <button
        type="button"
        onClick={onSegment}
        disabled={
          isLoading || !apiStatus.connected || selectedFeatures.length === 0
        }
        className="btn btn-primary w-full"
      >
        {isLoading ? "Processing..." : "Run Segmentation"}
      </button>
    </div>
  );
};

export default SegmentationForm;
