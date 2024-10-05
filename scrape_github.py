import requests
from requests.auth import HTTPBasicAuth

# Replace these with your GitHub username and personal access token
GITHUB_USERNAME = "PaliC"
GITHUB_TOKEN = "XXXXX"


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
    search_results = search_github_code(query, per_page=100, page=20)

    for item in search_results.get("items", []):
        repo_name = item["repository"]["full_name"]
        file_path = item["path"]
        print(f"\nFetching content from {repo_name}/{file_path}")
        try:
            file_content = get_file_content(repo_name, file_path)
            print(file_content[:500])  # Print the first 500 characters of the file
            break
        except Exception as e:
            print(f"Error fetching file content: {e}")

if __name__ == "__main__":
    main()
