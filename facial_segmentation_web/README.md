# Facial Segmentation Web App

A modern web application for facial feature segmentation using React, Tailwind CSS, and PyTorch models.

## Features

- **Modern UI**: Clean and responsive interface built with React and Tailwind CSS
- **Camera Integration**: Take photos directly from your webcam
- **Upload Images**: Upload images from your device
- **Model Selection**: Choose which facial features to segment (eyes, nose, lips)
- **Real-time Visualization**: See segmentation results immediately
- **Output Options**: View color overlay or binary mask output

## Architecture

This project uses a client-server architecture:

- **Frontend**: React, Tailwind CSS, Vite (for fast development)
- **Backend**: Flask API that serves the PyTorch segmentation models
- **Models**: Pre-trained segmentation models for eyes, nose, and lips

## Getting Started

### Prerequisites

- Node.js (v14+)
- Python 3.8+
- PyTorch
- Webcam (for camera functionality)

### Installation

1. Clone this repository
2. Install frontend dependencies:
   ```
   cd facial_segmentation_web/frontend
   npm install
   ```
3. Install backend dependencies:
   ```
   cd ../backend
   pip install -r requirements.txt
   ```

### Running the application

1. Start the backend server:
   ```
   cd backend
   python app.py
   ```
2. Start the frontend development server:
   ```
   cd ../frontend
   npm run dev
   ```
3. Open your browser to http://localhost:5173

## Deployment

The app can be deployed as:

1. A static website + API server
2. A containerized application using Docker
3. A desktop application using Electron

## License

MIT License

## Credits

Models trained on the CelebAMask-HQ dataset.
