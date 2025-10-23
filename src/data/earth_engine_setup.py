"""
Google Earth Engine Authentication and Setup Guide

This module provides utilities for setting up and authenticating with
Google Earth Engine API for satellite imagery access.
"""

import ee
import os
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarthEngineAuth:
    """Handles Google Earth Engine authentication"""
    
    def __init__(self):
        """Initialize Earth Engine authentication"""
        self.authenticated = False
        self.service_account_key = None
    
    def authenticate_with_service_account(self, key_path: str) -> bool:
        """
        Authenticate using service account key
        
        Args:
            key_path: Path to service account JSON key file
            
        Returns:
            True if authentication successful
        """
        try:
            if not os.path.exists(key_path):
                logger.error(f"Service account key not found: {key_path}")
                return False
            
            # Read service account email from key file
            with open(key_path, 'r') as f:
                key_data = json.load(f)
                service_account = key_data.get('client_email')
            
            if not service_account:
                logger.error("Invalid service account key file")
                return False
            
            # Authenticate
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials)
            
            self.authenticated = True
            self.service_account_key = key_path
            logger.info(f"Successfully authenticated with service account: {service_account}")
            return True
            
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            return False
    
    def authenticate_interactive(self) -> bool:
        """
        Authenticate interactively (for development)
        
        Returns:
            True if authentication successful
        """
        try:
            # Try to authenticate (will open browser for first-time auth)
            ee.Authenticate()
            ee.Initialize()
            
            self.authenticated = True
            logger.info("Successfully authenticated interactively")
            return True
            
        except Exception as e:
            logger.error(f"Interactive authentication failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if authenticated"""
        if not self.authenticated:
            return False
        
        try:
            # Test authentication by making a simple query
            ee.Number(1).getInfo()
            return True
        except Exception:
            self.authenticated = False
            return False
    
    @staticmethod
    def setup_instructions():
        """Print setup instructions for Google Earth Engine"""
        instructions = """
╔════════════════════════════════════════════════════════════════════════╗
║         Google Earth Engine Setup Instructions                         ║
╚════════════════════════════════════════════════════════════════════════╝

📋 STEP 1: Register for Google Earth Engine Access
   1. Go to: https://earthengine.google.com/
   2. Click "Get Started" or "Sign Up"
   3. Sign in with your Google account
   4. Complete the registration form
   5. Wait for approval email (usually within 24-48 hours)

📋 STEP 2: Option A - Interactive Authentication (Development)
   1. Run: python src/data/earth_engine_setup.py --interactive
   2. Follow the browser prompts to authenticate
   3. Copy the authentication code when prompted

📋 STEP 3: Option B - Service Account (Production)
   1. Go to: https://console.cloud.google.com/
   2. Create a new project or select existing project
   3. Enable Earth Engine API:
      - Go to "APIs & Services" > "Library"
      - Search for "Earth Engine API"
      - Click "Enable"
   
   4. Create Service Account:
      - Go to "IAM & Admin" > "Service Accounts"
      - Click "Create Service Account"
      - Name: "earth-engine-service"
      - Click "Create and Continue"
      - Grant role: "Earth Engine Resource Admin"
      - Click "Done"
   
   5. Create Key:
      - Click on the service account you created
      - Go to "Keys" tab
      - Click "Add Key" > "Create new key"
      - Select "JSON" format
      - Click "Create" (key will download)
   
   6. Save Key:
      - Move the downloaded JSON file to:
        config/earth-engine-service-account.json
      - Update .env file:
        GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT_KEY=./config/earth-engine-service-account.json

📋 STEP 4: Register Service Account with Earth Engine
   1. Go to: https://code.earthengine.google.com/
   2. Click on "Assets" tab
   3. Register your service account email (from JSON file)
   4. Wait for registration confirmation

📋 STEP 5: Test Authentication
   Run: python src/data/earth_engine_setup.py --test

🔗 Useful Links:
   - Earth Engine Signup: https://earthengine.google.com/signup/
   - Documentation: https://developers.google.com/earth-engine
   - Python API: https://developers.google.com/earth-engine/guides/python_install
   - Service Accounts: https://developers.google.com/earth-engine/guides/service_account

⚠️  IMPORTANT NOTES:
   - Free tier has usage limits (check quotas regularly)
   - Service account registration may take 24-48 hours
   - Keep your service account key secure (never commit to git)
   - Add service account key path to .gitignore

💡 Quick Start for Development:
   If you just want to test quickly, use interactive authentication:
   
   from src.data.earth_engine_setup import EarthEngineAuth
   auth = EarthEngineAuth()
   auth.authenticate_interactive()
   
════════════════════════════════════════════════════════════════════════════
"""
        print(instructions)


def test_authentication():
    """Test Earth Engine authentication"""
    try:
        # Try to get a simple result
        point = ee.Geometry.Point([72.8777, 19.0760])
        value = ee.Number(1).add(1)
        result = value.getInfo()
        
        logger.info(f"✓ Authentication test successful! Result: {result}")
        logger.info(f"✓ Earth Engine is ready to use")
        
        # Test image collection access
        collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        count = collection.filterBounds(point).filterDate('2024-01-01', '2024-01-07').size().getInfo()
        logger.info(f"✓ Can access Sentinel-2 cloud probability data (found {count} images)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Authentication test failed: {e}")
        return False


def main():
    """Main function for Earth Engine setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Earth Engine Setup')
    parser.add_argument('--interactive', action='store_true', 
                       help='Authenticate interactively')
    parser.add_argument('--service-account', type=str,
                       help='Path to service account JSON key')
    parser.add_argument('--test', action='store_true',
                       help='Test authentication')
    parser.add_argument('--instructions', action='store_true',
                       help='Show setup instructions')
    
    args = parser.parse_args()
    
    auth = EarthEngineAuth()
    
    if args.instructions:
        EarthEngineAuth.setup_instructions()
        return
    
    if args.interactive:
        logger.info("Starting interactive authentication...")
        if auth.authenticate_interactive():
            logger.info("✓ Authentication successful!")
            test_authentication()
        else:
            logger.error("✗ Authentication failed")
            EarthEngineAuth.setup_instructions()
    
    elif args.service_account:
        logger.info(f"Authenticating with service account: {args.service_account}")
        if auth.authenticate_with_service_account(args.service_account):
            logger.info("✓ Authentication successful!")
            test_authentication()
        else:
            logger.error("✗ Authentication failed")
            EarthEngineAuth.setup_instructions()
    
    elif args.test:
        logger.info("Testing authentication...")
        # Try to initialize (assumes already authenticated)
        try:
            ee.Initialize()
            auth.authenticated = True
            test_authentication()
        except Exception as e:
            logger.error(f"Not authenticated: {e}")
            EarthEngineAuth.setup_instructions()
    
    else:
        # Show instructions by default
        EarthEngineAuth.setup_instructions()


if __name__ == "__main__":
    main()