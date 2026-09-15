import asyncio
from nemoguardrails import LLMRails, RailsConfig

async def main():
    config = RailsConfig.from_path("./guardrails")
    rails = LLMRails(config)

    print("\n--- Test 1: Out-of-Scope Query ---")
    response1 = await rails.generate_async(messages=[{"role": "user", "content": "What is the capital of France?"}])
    print(response1)

    print("\n--- Test 2: In-Scope Query ---")
    response2 = await rails.generate_async(messages=[{"role": "user", "content": "Hello"}])
    print(response2)

if __name__ == "__main__":
    asyncio.run(main())