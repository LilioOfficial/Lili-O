from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="Qwen/Qwen3-14B-GGUF",
	filename="Qwen3-14B-Q4_K_M.gguf",
)
print("------------------------------")
abc = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print(abc)