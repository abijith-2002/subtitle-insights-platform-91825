#!/bin/bash
cd /home/kavia/workspace/code-generation/subtitle-insights-platform-91825/subtitle_frontend
npm run build
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
   exit 1
fi

