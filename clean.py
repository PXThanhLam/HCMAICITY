import os
root_path='/data/submission_output'
all_lines = []
list_file=os.listdir(root_path)
list_file=sorted(list_file)
print(list_file)
for f in list_file:
	if not f.endswith('txt'):
		continue
	if f[-10:-7] !='cam':
		continue
	f=root_path+'/'+f
	fi = open(f,'rb')
	lines = [li.decode("utf-8").strip("\n") for li in fi.readlines()]
	remove_lines = []
	for i, li in enumerate(lines):
		parts = li.split(',')
		if parts[2] =='' or parts[2]=='undetermine':
			remove_lines.append(li)
			continue
		if parts[3] =='' and parts[2]!='':
			parts[3] = '4'
			lines[i] = ','.join(parts)		
	for rli in remove_lines:
		lines.remove(rli)
	lines = sorted(lines, key=lambda line: int(line.split(',')[1]))
	all_lines +=lines

fo = open(root_path+'/submission.txt','w')
for li in all_lines:
	parts = li.strip('\n').strip('\r').split(',')
	li = ' '.join(parts)
	fo.write(li+'\n')
fo.close()



