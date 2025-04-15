import React from "react";
import { FaSmileBeam } from "react-icons/fa";

const Header = () => {
  return (
    <header className="border-b border-gray-200 bg-white shadow-sm">
      <div className="container mx-auto flex items-center justify-between px-4 py-4">
        <div className="flex items-center space-x-2">
          <FaSmileBeam className="h-8 w-8 text-primary-600" />
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              Facial Feature Segmentation
            </h1>
            <p className="text-sm text-gray-500">
              Eyes, Nose, and Lips Detection
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <a
            href="https://github.com/your-username/facial-segmentation-app"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-600 hover:text-primary-600"
          >
            GitHub
          </a>
          <span className="text-sm font-medium">v1.0.0</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
