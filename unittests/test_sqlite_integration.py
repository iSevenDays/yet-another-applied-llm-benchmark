#!/usr/bin/env python3
"""
Integration test for DockerJob SQLite fix.
This tests the exact scenario that was causing 99% hangs.
"""
import unittest
import subprocess
import time
import sys
import os

from docker_controller import DockerJob


class TestSQLiteIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment - check if Docker is available."""
        try:
            result = subprocess.run(['docker', 'version'], capture_output=True, check=True)
            cls.docker_available = True
            
            # Check if we have containers running
            result = subprocess.run(['docker', 'ps', '--format', '{{.ID}}\t{{.Image}}'], 
                                  capture_output=True, text=True)
            cls.container_id = None
            for line in result.stdout.strip().split('\n'):
                if line and 'llm-benchmark-image' in line:
                    cls.container_id = line.split('\t')[0]
                    break
                    
            if not cls.container_id:
                cls.skip_integration_tests = True
                print("WARNING: No llm-benchmark-image container found, skipping integration tests")
            else:
                cls.skip_integration_tests = False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            cls.docker_available = False
            cls.skip_integration_tests = True
            print("WARNING: Docker not available, skipping integration tests")
    
    def test_sqlite_hang_fix(self):
        """Test that SQLite no longer hangs with our enhanced DockerJob."""
        if self.skip_integration_tests:
            self.skipTest("Docker not available or no test container")
            
        container_id = self.container_id
        
        # Create the SQLite database first
        setup_cmd = ['docker', 'exec', container_id, 'python3', '-c', '''
import sqlite3
conn = sqlite3.connect("people.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS people (a_name TEXT, favorite_friend TEXT, person_age INT, the_profession TEXT)")
cursor.executemany("INSERT OR REPLACE INTO people VALUES (?,?,?,?)", [
    ("Alice", "Bob", 29, "Engineer"),
    ("Bob", "Alice", 32, "Doctor"), 
    ("Eve", "Frank", 25, "Teacher")
])
conn.commit()
conn.close()
print("Database setup complete")
''']
        
        setup_result = subprocess.run(setup_cmd, capture_output=True, text=True, timeout=10)
        print(f"Setup result: {setup_result.stdout}")
        if setup_result.stderr:
            print(f"Setup stderr: {setup_result.stderr}")
        
        try:
            # Now test our enhanced DockerJob
            start_time = time.time()
            job = DockerJob(container_id, "sqlite>")
            
            # This was the command that caused 15+ second hangs before our fix
            result = job("sqlite3 people.db")
            
            duration = time.time() - start_time
            
            # Clean up the job
            job.process.terminate()
            
            # Assertions for test validation
            self.assertLess(duration, 10, 
                           f"SQLite should complete quickly, took {duration:.1f}s (vs 15+ second hang before)")
            self.assertIsInstance(result, str, "Should return a string result")
            
        except subprocess.TimeoutExpired:
            self.fail("SQLite integration test timed out")
        except Exception as e:
            self.fail(f"SQLite integration test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)