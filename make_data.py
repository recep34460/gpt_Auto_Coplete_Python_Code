import os
import time
import tqdm
MAX_CHAR_LENGTH =512
MIN_CHAR_LENGHT = 256
NEWLINECHAR = "<N>"
full_paths=[]

for dirpath, dirnames, filenames in os.walk("repos"):
    
    for f in filenames:

        full_path = os.path.join(dirpath,f)
        full_paths.append(full_path)

print(len(full_paths))


with open("python_code_text_data.txt","a")as f:
    for fpath in full_paths:
        try:
            d = open(fpath,"r").read()
            #print(d)
            fd = d.replace("\n", NEWLINECHAR)
            if 97 < len(d) <= MAX_CHAR_LENGTH:

                
                f.write(fd+'\n')
                
            else:
                sd = fd.split(f"{NEWLINECHAR}{NEWLINECHAR}")
                substring = ""
                for split in sd:
                    substring += split+f"{NEWLINECHAR}{NEWLINECHAR}"
                    if MIN_CHAR_LENGHT <= len(substring) <= MAX_CHAR_LENGTH:
                        f.write(substring+'\n')
                        substring=""
        except Exception as e:
            print(str(e))    


        