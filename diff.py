from difflib import SequenceMatcher
text1 = open('MBSet.cu').read()
text2 = open('MBSet.cc').read()
m = SequenceMatcher(None, text1, text2)
print m.ratio()