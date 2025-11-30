"""
Test script for FastAPI churn prediction service.

Usage:
    1. Start the API: uvicorn api.app:app --reload
    2. Run this script: python scripts/test_api.py
"""

import requests
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test single customer prediction."""
    print("Testing /predict endpoint...")
    
    customer_data = {
        "Age": 35,
        "Gender": "Female",
        "Tenure": 18,
        "Usage_Frequency": 12,
        "Support_Calls": 8,
        "Payment_Delay": 15,
        "Subscription_Type": "Standard",
        "Contract_Length": "Monthly",
        "Total_Spend": 540.0,
        "Last_Interaction": 8,
        "TenureBucket": "1-2y",
        "HighSupport": 1,
        "RecentInteraction": 1,
        "UsagePerTenure": 0.67,
        "SpendPerMonth": 30.0,
    }
    
    response = requests.post(f"{API_URL}/predict", json=customer_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_prediction():
    """Test batch prediction."""
    print("Testing /predict/batch endpoint...")
    
    batch_data = {
        "customers": [
            {
                "Age": 25,
                "Gender": "Male",
                "Tenure": 6,
                "Usage_Frequency": 5,
                "Support_Calls": 2,
                "Payment_Delay": 5,
                "Subscription_Type": "Basic",
                "Contract_Length": "Monthly",
                "Total_Spend": 180.0,
                "Last_Interaction": 3,
                "TenureBucket": "<1y",
                "HighSupport": 0,
                "RecentInteraction": 1,
                "UsagePerTenure": 0.83,
                "SpendPerMonth": 30.0,
            },
            {
                "Age": 45,
                "Gender": "Female",
                "Tenure": 48,
                "Usage_Frequency": 20,
                "Support_Calls": 1,
                "Payment_Delay": 0,
                "Subscription_Type": "Premium",
                "Contract_Length": "Annual",
                "Total_Spend": 2400.0,
                "Last_Interaction": 2,
                "TenureBucket": "2-4y",
                "HighSupport": 0,
                "RecentInteraction": 1,
                "UsagePerTenure": 0.42,
                "SpendPerMonth": 200.0,
            },
        ]
    }
    
    response = requests.post(f"{API_URL}/predict/batch", json=batch_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_model_info():
    """Test model info endpoint."""
    print("Testing /model/info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("API TESTING SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API")
        print("Make sure the API is running: uvicorn api.app:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")

