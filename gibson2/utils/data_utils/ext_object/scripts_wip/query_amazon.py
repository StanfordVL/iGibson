from test_amazon_new_api import getDimsAndWeight
import json


# define the name of the file to read from
inputCatsFile = "toProcess.txt"
finalOutFile = "C:\\Users\\gokul\\Documents\\GitHub\\ig_dataset\\objects\\avg_category_specs.json"
workFile = "specs.json"


def findNewCats():
    with open(finalOutFile, 'r') as f:
        processedData = json.load(f)

    # open the file for reading
    filehandle = open(inputCatsFile, 'r')

    cats = set()

    while True:
        # read a single line
        line = filehandle.readline()
        if not line or line == "":
            break
        cat = line.split()[0]
        if cat not in processedData:
            cats.add(cat)

    filehandle.close()

    obj_list = list(cats)
    if len(obj_list):
        print("New categories: ")
        print(obj_list)

    return obj_list

def processNewCats():
    obj_list = findNewCats()
    specs = getDimsAndWeight(obj_list)

    with open(workFile, 'w') as outfile:
        json.dump(specs, outfile, indent = 4)
        print("Dumped data to " + workFile)

def updateDensities():
    with open(workFile, 'r') as f:
        data = json.load(f)
    for k in data:
        data[k]['density'] = data[k]['mass'] / data[k]['size'][0] / data[k]['size'][1] / data[k]['size'][2]

    with open(workFile, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

def writeToFinal():
    updateDensities()
    with open(workFile, 'r') as f:
        data = json.load(f)
    with open(finalOutFile, 'r') as f:
        processedData = json.load(f)
    processedData.update(data)
    with open(finalOutFile, 'w') as outfile:
        json.dump(processedData, outfile, indent = 4)
    print("Updated final output file")

def sanityCheck():
    obj_list = findNewCats()
    if len(obj_list):
        print("Missing Categories: ")
        print(obj_list)
        return
    print("Sanity Check passed")

#for d in data:
    #if not data[d]["size"][0] >= data[d]["size"][1]:
    #    print(d)
#    assert(data[d]["size"][0] >= data[d]["size"][1])
# outF = open("modelToSynsetToCat.txt", "w")
# for line in toWrite:
#   # write line to output file
#   outF.write(line)
#   outF.write("\n")
# outF.close()
