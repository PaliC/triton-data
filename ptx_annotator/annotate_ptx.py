# use anthropic api to annotate ptx code of a given file

import os
import anthropic
import toml
import dotenv

# get anthropic api key from .env file
dotenv.load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
ptx_code_file_path = "example.ptx"

client = anthropic.Client(api_key=anthropic_api_key)

# read prompt from prompts.toml file
with open("prompts.toml", "r") as file:
    prompts = toml.load(file)
for prompt in prompts["prompts"]:
    if prompt["name"] == "ptx_annotator":   
        ptx_annotator_prompt = prompt["prompt"]
        break


with open(ptx_code_file_path, "r") as file:
    ptx_code = file.read()

ptx_annotator_prompt = ptx_annotator_prompt.replace("{{PTX_CODE}}", ptx_code)


response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": ptx_annotator_prompt}],
    max_tokens=4096,
)

print(response.content[0].text)
