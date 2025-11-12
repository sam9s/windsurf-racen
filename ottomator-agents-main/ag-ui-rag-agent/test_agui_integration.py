#!/usr/bin/env python3
"""
Test script for AGUI-enabled RAG Agent integration.

This script validates that the agent is properly set up with AGUI support
and can handle the shared state correctly.
"""

import asyncio
import json
from typing import Dict, Any
import httpx
from pydantic import BaseModel
from datetime import datetime


class TestResult(BaseModel):
    """Test result model."""
    test_name: str
    status: str
    message: str
    timestamp: str


async def test_agui_endpoint(base_url: str = "http://localhost:8000") -> TestResult:
    """Test if the AGUI endpoint is accessible."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")

            if response.status_code == 200:
                return TestResult(
                    test_name="AGUI Endpoint",
                    status="PASS",
                    message="AGUI endpoint is accessible",
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="AGUI Endpoint",
                    status="FAIL",
                    message=f"Unexpected status code: {response.status_code}",
                    timestamp=datetime.now().isoformat()
                )
    except Exception as e:
        return TestResult(
            test_name="AGUI Endpoint",
            status="FAIL",
            message=f"Connection error: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


async def test_agui_state_sync(base_url: str = "http://localhost:8000") -> TestResult:
    """Test AGUI state synchronization."""
    try:
        async with httpx.AsyncClient() as client:
            # Send a test AGUI request
            test_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test message for state sync"
                    }
                ],
                "state": {
                    "retrieved_chunks": [],
                    "current_query": None,
                    "search_history": [],
                    "selected_chunk_id": None,
                    "total_chunks_in_kb": 0,
                    "knowledge_base_status": "testing"
                }
            }

            response = await client.post(
                f"{base_url}/run",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                return TestResult(
                    test_name="State Synchronization",
                    status="PASS",
                    message="State sync request successful",
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="State Synchronization",
                    status="FAIL",
                    message=f"State sync failed: {response.status_code}",
                    timestamp=datetime.now().isoformat()
                )
    except Exception as e:
        return TestResult(
            test_name="State Synchronization",
            status="FAIL",
            message=f"Error during state sync: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


async def test_search_tool(base_url: str = "http://localhost:8000") -> TestResult:
    """Test the search_knowledge_base tool."""
    try:
        async with httpx.AsyncClient() as client:
            # Test search functionality through AGUI
            search_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for test information in the knowledge base"
                    }
                ],
                "state": {
                    "retrieved_chunks": [],
                    "current_query": None,
                    "search_history": [],
                    "selected_chunk_id": None,
                    "total_chunks_in_kb": 0,
                    "knowledge_base_status": "ready"
                },
                "tools": ["search_knowledge_base"]
            }

            response = await client.post(
                f"{base_url}/run",
                json=search_payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )

            if response.status_code == 200:
                # Check if response contains events
                response_text = response.text
                if "event:" in response_text or "data:" in response_text:
                    return TestResult(
                        test_name="Search Tool",
                        status="PASS",
                        message="Search tool executed successfully",
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    return TestResult(
                        test_name="Search Tool",
                        status="WARNING",
                        message="Search executed but no events returned",
                        timestamp=datetime.now().isoformat()
                    )
            else:
                return TestResult(
                    test_name="Search Tool",
                    status="FAIL",
                    message=f"Search tool failed: {response.status_code}",
                    timestamp=datetime.now().isoformat()
                )
    except Exception as e:
        return TestResult(
            test_name="Search Tool",
            status="FAIL",
            message=f"Error testing search tool: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


async def test_database_connection() -> TestResult:
    """Test database connectivity."""
    try:
        from agent.dependencies import AgentDependencies

        deps = AgentDependencies()
        await deps.initialize()

        # Test a simple query
        async with deps.db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")

        await deps.cleanup()

        return TestResult(
            test_name="Database Connection",
            status="PASS",
            message=f"Database connected, {result} chunks found",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return TestResult(
            test_name="Database Connection",
            status="FAIL",
            message=f"Database connection failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("AGUI RAG Agent Integration Tests")
    print("="*60 + "\n")

    tests = [
        test_database_connection(),
        test_agui_endpoint(),
        test_agui_state_sync(),
        test_search_tool(),
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    passed = 0
    failed = 0
    warnings = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Test Error: {str(result)}")
            failed += 1
        else:
            icon = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
            print(f"{icon} {result.test_name}: {result.status}")
            print(f"   {result.message}")

            if result.status == "PASS":
                passed += 1
            elif result.status == "FAIL":
                failed += 1
            else:
                warnings += 1

    print("\n" + "="*60)
    print(f"Test Summary: {passed} passed, {failed} failed, {warnings} warnings")
    print("="*60 + "\n")

    if failed == 0:
        print("✨ All critical tests passed! The AGUI integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and ensure:")
        print("  1. The database is running and accessible")
        print("  2. The AGUI agent server is running on port 8000")
        print("  3. Environment variables are properly configured")


if __name__ == "__main__":
    asyncio.run(run_all_tests())