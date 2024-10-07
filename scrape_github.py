import requests
from requests.auth import HTTPBasicAuth
import tiktoken
import time

# Replace these with your GitHub username and personal access token
GITHUB_USERNAME = "PaliC"
GITHUB_TOKEN = "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"

def search_github_code(query, per_page=10, page=1):
    url = f"https://api.github.com/search/code?q={query}&per_page={per_page}&page={page}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(GITHUB_USERNAME, GITHUB_TOKEN))
    print(response.json())
    response.raise_for_status()
    return response.json()

def get_file_content(repo_full_name, path):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(GITHUB_USERNAME, GITHUB_TOKEN))
    response.raise_for_status()
    content = response.json()
    # The file content is base64 encoded, decode it
    import base64
    file_content = base64.b64decode(content['content']).decode('utf-8')
    return file_content

def main():
    # Search for files containing '@triton.jit'
    query = "@triton.jit+in:file+language:python"
    max_files = 1000
    total_tokens = 0
    total_files = 0
    page = 100
    while total_files < max_files:
        search_results = search_github_code(query, per_page=100, page=page)
        if not search_results.get("items"):
            break
        for item in search_results["items"]:
            repo_name = item["repository"]["full_name"]
            file_path = item["path"]
            try:
                file_content = get_file_content(repo_name, file_path)
                # Tokenize the file content
                encoding = tiktoken.encoding_for_model("gpt-4o")
                tokens = encoding.encode(file_content)
                total_tokens += len(tokens)
                total_files += 1
                # dump the file content to a file
                with open(f"github_triton_data/file_{total_files}.txt", "w") as f:
                    f.write(file_content)
                if total_files % 100 == 0:
                    print(f"Total tokens: {total_tokens}")
                    print(f"Total files: {total_files}")
                    print(f"Average tokens per file: {total_tokens / total_files}")
                if total_files >= 1000: 
                    break
            except Exception as e:
                print(f"Error fetching file content: {e}")
        # create a 1 minute sleep
        time.sleep(60)
        page += 1
    print(f"Total tokens: {total_tokens}")
    print(f"Total files: {total_files}")
    print(f"Average tokens per file: {total_tokens / total_files}")
if __name__ == "__main__":
    main()
