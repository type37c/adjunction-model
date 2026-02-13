#!/bin/bash
# Prepare adjunction-model code for Kaggle dataset upload

set -e

echo "Preparing adjunction-model code for Kaggle..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

# Copy necessary files
echo "Copying files..."
cp -r src/ $TEMP_DIR/
cp -r experiments/ $TEMP_DIR/
cp -r data/ $TEMP_DIR/ 2>/dev/null || echo "No data/ directory"
cp requirements.txt $TEMP_DIR/ 2>/dev/null || echo "No requirements.txt"
cp README.md $TEMP_DIR/

# Create archive
echo "Creating archive..."
cd $TEMP_DIR
zip -r adjunction-model-code.zip . -x "*.pyc" -x "__pycache__/*" -x "*.git/*"

# Move to kaggle directory
mv adjunction-model-code.zip /home/ubuntu/adjunction-model/kaggle/

# Cleanup
cd /home/ubuntu/adjunction-model
rm -rf $TEMP_DIR

echo "✓ Archive created: kaggle/adjunction-model-code.zip"
echo ""
echo "Next steps:"
echo "1. Go to https://www.kaggle.com/datasets"
echo "2. Click 'New Dataset'"
echo "3. Upload kaggle/adjunction-model-code.zip"
echo "4. Title: 'Adjunction Model Code'"
echo "5. Make it private"
