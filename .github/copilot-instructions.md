<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	<!-- Cloud Burst Prediction System - Phase 1 MVP with meteorological data ingestion, satellite imagery processing, ML models, and web dashboard. -->

- [x] Scaffold the Project
	<!--
	✅ Complete project structure created with:
	- Source code modules (data, preprocessing, features, models, api, dashboard)
	- Configuration files (config.yaml, .env.example)
	- Main pipeline script (run_pipeline.py)
	- Requirements.txt with all dependencies
	- README.md with comprehensive documentation
	- All necessary directories and __init__.py files
	-->

- [x] Customize the Project
	<!--
	✅ Project fully customized for Cloud Burst Prediction System with:
	- Complete weather API integration (Open-Meteo, OpenWeatherMap)
	- Satellite imagery processing using Google Earth Engine
	- Advanced image processing with OpenCV and GLCM features
	- Comprehensive feature engineering with atmospheric indices
	- Baseline ML models: Random Forest, SVM, LSTM
	- FastAPI backend with prediction endpoints
	- Interactive Streamlit dashboard with maps and charts
	- Full pipeline orchestration script
	- Unit tests and development documentation
	-->

- [ ] Install Required Extensions
	<!-- Python extensions and ML/data science extensions will be installed. -->

- [ ] Compile the Project
	<!--
	Setup Python environment and install dependencies.
	-->

- [ ] Create and Run Task
	<!--
	Create tasks for data pipeline, model training, and dashboard launch.
	-->

- [ ] Launch the Project
	<!--
	Launch development environment and dashboard.
	-->

- [ ] Ensure Documentation is Complete
	<!--
	Complete README.md and project documentation.
	-->

## Project-Specific Instructions

This is a Cloud Burst Prediction System that uses:
- Weather API data ingestion (Open-Meteo, OpenWeatherMap)
- Satellite imagery processing (Google Earth Engine)
- Machine learning models (Random Forest, SVM, LSTM)
- FastAPI backend and Streamlit dashboard
- OpenCV for image processing
- Feature engineering for meteorological data

Key architectural components:
- `src/data/` - Data ingestion and storage
- `src/preprocessing/` - Data cleaning and transformation
- `src/features/` - Feature engineering
- `src/models/` - ML model implementations
- `src/api/` - FastAPI endpoints
- `src/dashboard/` - Streamlit dashboard
- `config/` - Configuration files
- `notebooks/` - Jupyter notebooks for experimentation