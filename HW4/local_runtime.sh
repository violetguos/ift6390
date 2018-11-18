#!/usr/bin/env bash
#chmod +x vk.sh

# to start a local runtime for Google Collab
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8887 \
  --NotebookApp.port_retries=0
