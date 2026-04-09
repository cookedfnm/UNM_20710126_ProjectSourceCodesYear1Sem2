#!/bin/bash
# Run this once on your Pi to set up the template directory.
# Place your template images in ~/pw3/final/templates/

TEMPLATE_DIR="$HOME/pw3/final/templates"
mkdir -p "$TEMPLATE_DIR"

echo "Copy your 16 template images into: $TEMPLATE_DIR"
echo "Expected files:"
echo "  3_4circle.png  blue_arrow_right.jpg  circular_segment.png  diamond.png"
echo "  fingerprint.png  green_arrow_up.jpg  octagon.png  orange_arrow_left.png"
echo "  plus.png  press_button.png  qr_code.png  recycle.png"
echo "  red_arrow_down.png  star.png  trapezoid.png  warning.png"
echo ""
echo "Then run:  python3 main.py"
