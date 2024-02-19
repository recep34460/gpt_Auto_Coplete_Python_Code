import os
#from curtsies.fmtfuncs import red,bold, green, on_blue,yellow,blue,cyan
import time
from tqdm import tqdm

d ="repos"

for dirpath, dirnames, filenames in tqdm(os.walk(d)):
    
    for f in filenames:

        full_path = os.path.join(dirpath,f)
        #print(os.path.join(dirpath, filenames))
        #print(full_path)
        if full_path.endswith(".py"):
            #print(f"Keeping {full_path}")
            pass
        else:
            #print(f"Deleting{full_path}")
            if d in full_path:
                os.chmod(full_path, 0o777)
                os.remove(full_path)
            else:
                print("something is wrong")
                time.sleep(60)
            
    #time.sleep(0.5)
    #break
