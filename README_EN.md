# KinetiQ

KinetiQ is a sports rehabilitation assessment app built with Streamlit. It uses MediaPipe pose landmarks for local posture analysis, then sends structured findings to DeepSeek to generate therapist-facing rehabilitation reports.

## Features

- Front and side photo upload
- Pose landmark detection and annotated pose images
- ACL risk stratification and kinetic chain analysis
- Muscle function hypothesis mapping
- DeepSeek-powered personalized rehab recommendations
- Patient archive with report history
- HTML and PDF report export

## Project Structure

- `main.py`: Streamlit UI
- `app_pipeline.py`: end-to-end assessment pipeline
- `pose.py`: MediaPipe pose detection and annotated image generation
- `analysis.py`: pose metrics, ACL risk, kinetic chain summary
- `clinical_knowledge.py`: muscle mapping and report template
- `deepseek_client.py`: DeepSeek API client
- `records_store.py`: patient archive storage
- `report_export.py`: HTML and PDF export helpers

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Place `pose_landmarker.task` in the project root.
4. Set `DEEPSEEK_API_KEY` if you want AI-generated reports.

## Run

```bash
streamlit run main.py
```

## Notes

- If `DEEPSEEK_API_KEY` is not set, the app falls back to a local therapist-style report template.
- Patient records and uploaded images are stored locally under `data/`.
- This project is for screening and workflow support, not medical diagnosis.
