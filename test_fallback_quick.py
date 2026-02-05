"""
Quick test script to verify fallback tracker functionality.
"""

from llm_adapter.fallback_tracker import get_fallback_tracker, FallbackEvent
from datetime import datetime


def test_fallback_tracker():
    """Test the fallback tracker basic functionality."""
    
    print("Testing Fallback Tracker...")
    print("=" * 60)
    
    # Get tracker instance
    tracker = get_fallback_tracker()
    
    # Clear any existing data
    tracker.clear()
    
    # Record a successful fallback
    print("\n1. Recording successful fallback...")
    event1 = tracker.record_fallback(
        provider="gemini",
        original_location="asia-southeast1",
        fallback_location="us-central1",
        original_model="gemini-2.0-flash",
        fallback_model="gemini-2.0-flash",
        error_message="Model not found in region",
        fallback_duration_ms=250.5,
        success=True
    )
    print(f"   ✓ Event recorded: {event1.timestamp}")
    
    # Record a failed fallback
    print("\n2. Recording failed fallback...")
    event2 = tracker.record_fallback(
        provider="gemini",
        original_location="europe-west1",
        fallback_location="us-central1",
        original_model="gemini-2.0-flash",
        fallback_model="gemini-2.0-flash",
        error_message="Permission denied",
        fallback_duration_ms=180.3,
        success=False
    )
    print(f"   ✓ Event recorded: {event2.timestamp}")
    
    # Record another successful fallback
    print("\n3. Recording another successful fallback...")
    event3 = tracker.record_fallback(
        provider="gemini",
        original_location="asia-southeast1",
        fallback_location="us-central1",
        original_model="gemini-2.5-flash",
        fallback_model="gemini-2.5-flash",
        error_message="Region unavailable",
        fallback_duration_ms=320.7,
        success=True
    )
    print(f"   ✓ Event recorded: {event3.timestamp}")
    
    # Get statistics
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    
    stats = tracker.get_stats()
    summary = stats.get_summary()
    
    print(f"\nTotal fallbacks: {summary['total_fallbacks']}")
    print(f"Successful: {summary['successful_fallbacks']}")
    print(f"Failed: {summary['failed_fallbacks']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total duration: {summary['total_duration_ms']:.2f}ms")
    print(f"Average duration: {summary['average_duration_ms']:.2f}ms")
    
    # Get recent events
    print("\n" + "=" * 60)
    print("Recent Events")
    print("=" * 60)
    
    recent = tracker.get_recent_events(limit=5)
    for i, event in enumerate(recent, 1):
        print(f"\nEvent {i}:")
        print(f"  Time: {event.timestamp.strftime('%H:%M:%S')}")
        print(f"  Provider: {event.provider}")
        print(f"  Route: {event.original_location} → {event.fallback_location}")
        print(f"  Model: {event.original_model}")
        print(f"  Success: {'✓' if event.success else '✗'}")
        print(f"  Duration: {event.fallback_duration_ms:.2f}ms")
        print(f"  Error: {event.error_message}")
    
    # Verify calculations
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    expected_total = 3
    expected_success = 2
    expected_failed = 1
    expected_avg = (250.5 + 180.3 + 320.7) / 3
    
    assert summary['total_fallbacks'] == expected_total, "Total count mismatch"
    assert summary['successful_fallbacks'] == expected_success, "Success count mismatch"
    assert summary['failed_fallbacks'] == expected_failed, "Failed count mismatch"
    assert abs(summary['average_duration_ms'] - expected_avg) < 0.01, "Average duration mismatch"
    
    print("\n✓ All assertions passed!")
    print("✓ Fallback tracker is working correctly!")
    
    # Clean up
    tracker.clear()
    print("\n✓ Tracker cleared")


if __name__ == "__main__":
    test_fallback_tracker()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
