#!/bin/bash
# Script to fix the UMAP callback automatically

echo "🔧 Fixing UMAP callback..."

# Make a backup first
cp sculpt/callbacks/umap_callbacks.py sculpt/callbacks/umap_callbacks.py.backup
echo "✅ Backup created: umap_callbacks.py.backup"

# Add missing imports after the existing imports
echo "📝 Adding missing imports..."
# Find the line with confidence_assessment import and add our imports after it
sed -i '' '/from sculpt.utils.metrics.confidence_assessment import (/,/)/a\
from sculpt.utils.metrics.physics_features import (\
    calculate_physics_features,\
    calculate_physics_features_with_profile,\
    calculate_physics_features_flexible,\
    has_physics_features,\
)' sculpt/callbacks/umap_callbacks.py

# Add missing State parameters to callback
echo "📝 Adding missing callback parameters..."
sed -i '' '/State("show-points-overlay", "value"),/a\
    State("file-config-assignments-store", "data"),\
    State("configuration-profiles-store", "data"),' sculpt/callbacks/umap_callbacks.py

# Add missing function parameters
echo "📝 Adding missing function parameters..."
sed -i '' '/show_points_overlay,$/a\
    assignments_store,\
    profiles_store,' sculpt/callbacks/umap_callbacks.py

echo "✅ Basic fixes applied!"
echo "⚠️  You still need to update the file processing logic manually"
echo "   Look for the file processing loop and add configuration profile support"

# Show what was changed
echo ""
echo "🔍 Changes made:"
echo "1. Added physics features imports"
echo "2. Added missing State parameters to callback"
echo "3. Added missing function parameters"
echo ""
echo "📁 Backup saved as: sculpt/callbacks/umap_callbacks.py.backup"
