import re
import unicodedata
from tqdm import tqdm

def format_text(text):
    ft = text.encode().decode('unicode-escape') #double backslash to single
    ft = unicodedata.normalize('NFKD',ft).encode('ascii', 'ignore').decode('ascii')
    #turn special chars into the nearest ascii chars
    #special chars could be letter a with bar on top, i.e. ā, 
    #and it should be turned into a
    ft = re.sub(r'\n',' ',ft) #line break to space
    ft = re.sub(r'\ +',' ',ft) #remove redundant space
    ft = re.sub(r'\\+',' ',ft) #remove redundant backslashes
    #the use of backslash in this dataset seems in consistent
    #sometimes double backslash, sometimes triple or even quadraple
    #thus there could be redundant backslashes
    #see for example  review No.2282 (in python index)
    return ft
    
def format_dataset(dataset):
    print("Formatting Dataset:")
    result = [format_text(text) for text in tqdm(dataset)]
    return result
