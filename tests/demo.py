import asyncio
import time

from dotenv import load_dotenv

load_dotenv()


async def demo_scene_1_contradiction():
    """Scene 1: Contradiction detection (90 seconds)"""
    print("\n" + "=" * 60)
    print("SCENE 1: Contradiction Detection")
    print("=" * 60 + "\n")

    from agent.loop import run_agent

    agent_id = "demo-agent"

    # Plant a fact
    print("Step 1: Planting a fact...")
    resp, rid, _ = await run_agent(
        agent_id, "Just so you know, our team standup is every Wednesday at 10am.", None
    )
    print(f"Agent: {resp}\n")
    await asyncio.sleep(2)

    # Now contradict it
    print("Step 2: Contradicting the fact...")
    resp, rid, contradiction = await run_agent(
        agent_id, "By the way, the standup was moved to Thursday at 9am.", rid
    )

    if contradiction:
        print("\n[LIVE DEMO MOMENT]")
        print(
            f"Contradiction detected with confidence: {contradiction['confidence_score']:.0%}"
        )
        print(f"  Old: {contradiction['conflicts_with']}")
        print(f"  New: {contradiction['new_fact']}")
        print(f"  Explanation: {contradiction['explanation']}")

    print(f"\nAgent: {resp}\n")


async def demo_scene_2_session_recovery():
    """Scene 2: Session Recovery (60 seconds)"""
    print("\n" + "=" * 60)
    print("SCENE 2: Session Recovery")
    print("=" * 60 + "\n")

    from agent.loop import run_agent
    from sandbox.checkpoint import checkpoint, resume

    agent_id = "recovery-demo"

    print("Step 1: Starting a task...")
    resp, rid, _ = await run_agent(
        agent_id,
        "Research the top 5 AI memory papers from 2025 and summarize each.",
        None,
    )
    print(f"Agent: {resp[:200]}...\n")

    print("Step 2: Simulating agent interruption — checkpointing now...")
    await checkpoint(agent_id)
    print("[CHECKPOINT SAVED TO BLAXEL SANDBOX + REDIS]\n")

    await asyncio.sleep(2)

    print("Step 3: Resuming agent (simulating new session)...")
    restored = await resume(agent_id)
    if restored:
        print(f"[RESTORED] Task: {restored.working.task}")
        print(f"[RESTORED] Progress: {restored.working.progress_pct}%")
        print(
            f"[RESTORED] Resuming with {len(restored.recent_episodic)} memories loaded"
        )
    print()


async def demo_scene_3_confidence_decay():
    """Scene 3: Confidence decay visualization (45 seconds)"""
    print("\n" + "=" * 60)
    print("SCENE 3: Confidence Decay")
    print("=" * 60 + "\n")

    from memory.decay import calculate_current_confidence
    from memory.models import MemoryEntry

    # Show confidence of a memory over simulated time
    mem = MemoryEntry(
        id="demo",
        agent_id="demo",
        content="Meeting is at 3pm today",
        layer="episodic",
        source="user_said",
        decay_rate=0.05,
        created_at=time.time() - 7200,  # pretend it's 2 hours old
        last_reinforced=time.time() - 7200,
    )

    print("Simulating confidence decay for: 'Meeting is at 3pm today'")
    print("(decay_rate=0.05/hr, currently 2 hours old)\n")

    for hours in [0, 2, 6, 12, 24, 48, 72]:
        sim_time = mem.created_at + (hours * 3600)
        conf = calculate_current_confidence(mem, now=sim_time)
        bar = "█" * int(conf * 20)
        print(f"  {hours:3d}h: [{bar:<20}] {conf:.2f}")

    print("\nAt 0.1 threshold, memory is filtered from context window.")
    print("At 0.0, memory is pruned from storage entirely.\n")


if __name__ == "__main__":
    asyncio.run(demo_scene_1_contradiction())
    asyncio.run(demo_scene_2_session_recovery())
    asyncio.run(demo_scene_3_confidence_decay())

    print("\n" + "=" * 60)
    print("DASHBOARD: http://localhost:8000")
    print("Start with: uvicorn api.server:app --reload")
    print("=" * 60 + "\n")
