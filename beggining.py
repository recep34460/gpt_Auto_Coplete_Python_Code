#https://github.com/settings/tokens
#from curtsies.fmtfuncs import red, bold, green, on_blue,yellow,blue,cyan
from github import Github
import time
from github import Auth
from datetime import datetime
import os
end_time = 1614457443
start_time = end_time-86400
ACCESS_TOKEN = open("token.txt","r").read()
g = Github(ACCESS_TOKEN)
print(g.get_user())

for i in range(50):
    try:
        start_time_str = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d')
        end_time_str = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')
        query = f"flask language:python created:{start_time_str}..{end_time_str}"
        print(query)
        end_time -= 86400
        start_time -= 86400

        result = g.search_repositories(query)
        print(result.totalCount)
        
        for repository in result:
            print(f"{repository._clone_url}")
            
            print(f"{repository.clone_url}")
            print(f"{repository.tags_url}")
            print(f"{repository.owner.login}")
            os.system(f"git clone {repository.clone_url} repos/{repository.owner.login}/{repository.name}")
            #print(f"current start time {start_time}")
            #print(dir(repository))

    except Exception as e:
        print(str(e))
        print("Broke for some reason...")
        time.sleep(120)
print("finished",start_time)