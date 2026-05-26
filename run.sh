#!/bin/bash
# NOTE: This is run.sh instead of run.bat because this project runs on Ubuntu Linux.
# The assignment permits run.sh when run.bat is not possible (Linux environment).
# To run: open a terminal and type:  bash run.sh

cd "$(dirname "$0")"

echo "   Music Genre Classifier"

# Check model exists 
if [ ! -d "model/genre_model" ]; then
    echo "Model not found. Training now."
    echo "This will take 5-10 minutes on first run."
    python3 train.py Merged_dataset.csv model/genre_model
fi

# Check labels file exists 
if [ ! -f "data/labels.txt" ]; then
    echo "Error: data/labels.txt not found."
    echo "Please run: python3 train.py Merged_dataset.csv model/genre_model"
    exit 1
fi

echo ""
echo "Model found: model/genre_model"
echo "Labels: $(cat data/labels.txt | tr '\n' ' ')"
echo ""
echo "Open your browser and go to: http://localhost:5000"
echo ""

python3 music_classifier.py
