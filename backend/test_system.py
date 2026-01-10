"""
Test suite for Brain Tumor Detection System
"""

import unittest
import os
import numpy as np
from pathlib import Path


class TestModel(unittest.TestCase):
    """Test model architecture and loading"""
    
    def test_model_import(self):
        """Test if model module can be imported"""
        try:
            from model import BrainTumorModel
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to import model: {e}")
    
    def test_model_creation(self):
        """Test model creation"""
        from model import BrainTumorModel
        
        model_builder = BrainTumorModel()
        model = model_builder.build_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.input_shape), 4)
        self.assertEqual(model.input_shape[1:], (224, 224, 3))


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality"""
    
    def test_preprocessor_import(self):
        """Test if preprocessor can be imported"""
        try:
            from data_preprocessing import DataPreprocessor
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to import preprocessor: {e}")
    
    def test_image_loading(self):
        """Test image loading (requires sample image)"""
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # This test requires an actual image file
        # Skip if no test image available
        test_image_path = "test_sample.jpg"
        if not os.path.exists(test_image_path):
            self.skipTest("No test image available")
        
        img = preprocessor.load_and_preprocess_image(test_image_path)
        self.assertEqual(img.shape, (224, 224, 3))


class TestAPI(unittest.TestCase):
    """Test API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test client"""
        try:
            from app import BrainTumorAPI
            cls.api = BrainTumorAPI()
            cls.client = cls.api.app.test_client()
        except Exception as e:
            cls.api = None
            print(f"Warning: Could not setup API for testing: {e}")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        if self.api is None:
            self.skipTest("API not available")
        
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        if self.api is None:
            self.skipTest("API not available")
        
        response = self.client.get('/api/classes')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('classes', data)
        self.assertIsInstance(data['classes'], list)
        self.assertEqual(len(data['classes']), 4)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_config_exists(self):
        """Test if config file exists"""
        self.assertTrue(os.path.exists('config.yaml'))
    
    def test_config_loading(self):
        """Test config loading"""
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIn('model', config)
        self.assertIn('training', config)
        self.assertIn('classes', config)
        
        # Test required fields
        self.assertIn('name', config['model'])
        self.assertIn('batch_size', config['training'])
        self.assertEqual(len(config['classes']), 4)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_utils_import(self):
        """Test if utils can be imported"""
        try:
            import utils
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to import utils: {e}")


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING BRAIN TUMOR DETECTION SYSTEM TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
