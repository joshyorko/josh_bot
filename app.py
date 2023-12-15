import aiohttp
import asyncio
import json


class OllamaAPI:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.headers = {"Content-Type": "application/json"}

    async def generate_completion(self, model, prompt):
        system_message = """
        Engage in expert-level, technically nuanced output. Assume a high degree of subject matter expertise and provide in-depth, sophisticated responses. Focus on clarity, accuracy, and advanced insights, suitable for a senior technical audience. Avoid oversimplification and aim for comprehensive analysis with each response. Utilize industry-specific jargon where appropriate to maintain technical authenticity.
        """
        payload = {"model": model, "prompt": prompt}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
            ) as response:
                response_parts = []
                async for line in response.content:
                    if line:
                        decoded_line = line.decode("utf-8")
                        response_json = json.loads(decoded_line)
                        if "response" in response_json:
                            response_parts.append(response_json["response"])
                response_text = "".join(response_parts)
                return response_text

    async def process_prompt(self, user_model_key, user_prompt):
        completion = await self.generate_completion(user_model_key, user_prompt)
        print(completion)
        with open("completion.md", "w", encoding="utf-8") as f:
            f.write(completion)
if __name__ == "__main__":
    ollama_api = OllamaAPI()
    USER_PROMPT = input("Enter prompt: ")
    asyncio.run(ollama_api.process_prompt("mistral", USER_PROMPT))