# 🏓 PickleTrack - AI Sports Analytics Platform

**AI-Powered Pickleball Shot Tracking with Zone Accuracy Analytics**

PickleTrack is a computer vision system that tracks shot accuracy during pickleball practice sessions using YOLOv8 cone detection, automatic zone mapping, and real-time performance analytics.

## 🎯 Core Features

- **🤖 Cone Detection**: YOLOv8-based automatic cone detection and zone construction
- **📍 Zone Mapping**: Precise target zone definition from cone positions
- **⚡ Ball Tracking**: Real-time ball trajectory analysis and bounce point identification
- **📊 Shot Analysis**: Automatic classification of hits, misses, and zone accuracy
- **📈 Analytics Dashboard**: Professional Streamlit interface with interactive visualizations
- **💾 Data Export**: CSV/JSON export of detailed shot performance data

## 🚀 Quick Start

### 🎮 Try the Live Demo
```bash
# Run the interactive dashboard
streamlit run app.py
```
**✅ Demo mode included** - Experience PickleTrack with sample data immediately!

### 📦 Installation
```bash
# Clone and install
git clone <repository-url>
cd pickletrack
pip install -r requirements.txt
```

### 🎯 Process Your Videos
```bash
# Analyze your pickleball session
python main.py --video data/videos/your_session.mp4 --output data/outputs/
```

## 🏗️ Architecture

```
📽️ Video Input → 🤖 Cone Detection → 🎯 Zone Mapping → ⚡ Ball Tracking → 📊 Analysis → 📈 Dashboard
```

### Directory Structure
```
pickletrack/
├── src/                    # Core AI/ML modules
│   ├── detection/         # YOLOv8 ball and cone detection
│   ├── tracking/          # Ball trajectory tracking
│   ├── mapping/           # Court coordinate mapping
│   ├── analysis/          # Shot classification and metrics
│   └── visualization/     # Dashboard components
├── data/                  # Training data and models
├── web/                   # Portfolio and web interfaces
├── scripts/               # Utility scripts and demos
├── docs/                  # Additional documentation
├── app.py                # 🎯 Main Streamlit dashboard
└── main.py               # CLI processing pipeline
```

## 📊 Dashboard Features

### 🎯 Zone Accuracy Tracking (Core Feature)
- Real-time accuracy metrics for each target zone
- Color-coded performance indicators
- Shot distribution and comparison charts
- Professional coaching insights

### 📈 Performance Analytics
- Session summaries and key metrics
- Shot outcome distribution
- Court heatmaps and landing patterns
- Ball speed analysis and trends

### 📋 Data Management
- Detailed shot-by-shot breakdown
- Zone hit/miss tracking
- Export capabilities (CSV/JSON)
- Session comparison tools

## 🎥 Recording Guidelines

### 📹 Camera Setup
- **Position**: 7-10 ft high on target side of court
- **View**: Full target court coverage
- **Duration**: 2-3 minute sessions optimal
- **Shots**: 20-30 shots per video for best analysis

### 🎯 Target Setup
- Use bright, distinct cones for zone boundaries
- Define clear target areas (deep cross, middle, etc.)
- One shot type per video for consistent analysis

## 🛠️ Technology Stack

- **🤖 AI/ML**: YOLOv8 (Ultralytics), OpenCV, PyTorch
- **📊 Analytics**: NumPy, Pandas, Scikit-learn
- **📈 Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **🔧 Utilities**: Shapely (geometry), PyYAML (config)

## 🌐 Live Demo

**🔗 Try PickleTrack**: [Live Dashboard](https://pickletrack-app.streamlit.app)

Experience the full functionality with included demo data showcasing:
- 3 target zones with varying difficulty
- 15 training shots with realistic metrics
- Professional analytics and visualizations

## 🚀 Deployment

### Streamlit Cloud
```bash
# Deploy to Streamlit Cloud
streamlit run app.py
```

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/ tests/
flake8 src/ tests/
```

## 📈 Use Cases

### 🏓 For Players
- Track shot accuracy in practice drills
- Identify weak zones needing improvement
- Monitor progress over time
- Get data-driven performance insights

### 👨‍🏫 For Coaches
- Analyze player performance objectively
- Design targeted practice sessions
- Track improvement across sessions
- Provide visual feedback and recommendations

## 🔬 Technical Highlights

- **Custom YOLOv8 Training**: Specialized cone detection model
- **Computer Vision Pipeline**: Advanced ball tracking and trajectory analysis
- **Geometric Analysis**: Shapely-based zone construction and hit detection
- **Real-time Processing**: Efficient video analysis with progress tracking
- **Professional UI**: Streamlit dashboard with interactive visualizations

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- OpenCV for computer vision capabilities
- Streamlit for dashboard framework
- The pickleball community for inspiration and feedback

---

**🎯 PickleTrack: Where AI meets sports analytics for precision training and performance improvement.**