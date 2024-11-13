import requests
from requests.auth import HTTPBasicAuth
import tiktoken
import time
import os
import pprint
from dotenv import load_dotenv
import json
from typing import Optional, Dict, Any
from urllib.request import Request, urlopen

# get GITHUB_TOKEN from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN is None:
    print("No .env file found. Please copy .env.example to .env and set the GITHUB_TOKEN variable.")
    print("Instructions for getting a personal access token are here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens")
    exit(1)

def github_api_request(
    url: str,
    data: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Any:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token is not None:
        headers["Authorization"] = f"token {token}"

    _data = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=_data)) as conn:
            return json.load(conn)
    except Exception as err:
        print(f"Failed to get {url}: {err}")

def search_github_code(query, per_page=10, page=1, smallest_size=100, largest_size=10000):
    # wait for 10 seconds
    time.sleep(10)
    
    url = f"https://api.github.com/search/code?q={query}&per_page={per_page}&page={page}"

    return github_api_request(url, token=GITHUB_TOKEN)

def get_file_content(repo_full_name, path):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    content = github_api_request(url, token=GITHUB_TOKEN)
    # The file content is base64 encoded, decode it
    import base64
    file_content = base64.b64decode(content['content']).decode('utf-8')
    return file_content

def search_github_repos(query, per_page=100, page=1, smallest_size=100, largest_size=10000):
    url = f"https://api.github.com/search/repositories?q={query}&per_page={per_page}&page={page}"
    return github_api_request(url, token=GITHUB_TOKEN)

def get_repos_with_triton_mention():
    # Search for repos containing 'triton' or `pytorch` in the README, with more than 100 stars
    # repo_queries = ["triton+in:readme+language:python+stars:>100", "pytorch+in:readme+language:python+stars:>100"]
    repo_queries = ["triton+in:readme+language:python+stars:>100"]
    repos_with_triton_code = set()
    repo_names = set()
    counter = 0
    for repo_query in repo_queries:
        page = 1
        total_count = None
        while True:
            result = search_github_repos(repo_query, page=page)
            if total_count is None:
                total_count = result['total_count']
            repo_names.update([item["full_name"] for item in result["items"]])
            if len(result["items"]) < 100 or page * 100 >= total_count:
                break
            page += 1
    for repo in repo_names:
        print(f"search for {counter} of {len(repo_names)}")
        query = f"@triton.jit+in:file+language:python+repo:{repo}"
        result = search_github_code(query, per_page=1, page=1)
        count = result['total_count']
        print(f"there are {count} files found in {repo}")
        counter += 1
        if count > 0:
            repos_with_triton_code.add(repo)
        

    # save a file with the repos with triton code
    with open("repos_with_triton_code.txt", "w") as f:
        for repo in repos_with_triton_code:
            f.write(repo + "\n")
    return repos_with_triton_code

def main():

    if not os.path.exists("repos_with_triton_code.txt"):
        get_repos_with_triton_mention()
    # get the repos with triton code
    repos_with_triton_code = open("repos_with_triton_code.txt").read().splitlines()
    print(f"there are {len(repos_with_triton_code)} repos with triton code we will download them")
    # Create the 'downloads' folder if it doesn't exist
    if not os.path.exists("scraper/generator/paritybench_download"):
        os.makedirs("scraper/generator/paritybench_download")

    for repo in repos_with_triton_code:
        url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
        response = requests.get(url)
        repo_name = repo.replace("/", "_")  # Replace '/' with '_' for valid filename
        with open(os.path.join("scraper/generator/paritybench_download", f"{repo_name}.zip"), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {repo_name}.zip")

if __name__ == "__main__":
    main()
