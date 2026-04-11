import asyncio
from dotenv import load_dotenv

load_dotenv()

from agent.loop import interactive_session

if __name__ == "__main__":
    try:
        asyncio.run(interactive_session())
    except KeyboardInterrupt:
        print("\nExiting...")
