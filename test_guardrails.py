from nemoguardrails import LLMRails, RailsConfig

# Load guardrails configuration from the guardrails folder
config = RailsConfig.from_path("./guardrails")
rails = LLMRails(config)

# Test query 1: In-scope/Greeting
print("--- Test 1 ---")
response = rails.generate(messages=[{"role": "user", "content": "Hello"}])
print(response)

# Test query 2: Out-of-scope query
print("\n--- Test 2 ---")
response = rails.generate(messages=[{"role": "user", "content": "What is the weather today?"}])
print(response)