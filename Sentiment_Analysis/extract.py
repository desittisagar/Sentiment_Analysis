import csv
infile = open("text_emotion.csv","r")
reader = csv.reader(infile)
outfile = open("empty.csv","w")
writer = csv.writer(outfile)
c=0
for row in reader:
    r = row[1]
    if( r == "empty" ):
        string = []
        string.append(row[1])
        c += 1
		#string = row[1] + ','
        string.append(row[3])
        writer.writerow(string)
print(c)
infile.close()
outfile.close()		