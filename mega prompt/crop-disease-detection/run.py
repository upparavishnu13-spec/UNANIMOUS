#!/usr/bin/env python3
"""CropGuard AI - Crop Disease Detection Application

Run this script to start the application.
Usage: python run.py
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from app import create_app

def main():
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print(f"\n{'='*50}")
    print(f"  CropGuard AI - Crop Disease Detection")
    print(f"  Starting server on http://localhost:{port}")
    print(f"  Debug mode: {debug}")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    main()
