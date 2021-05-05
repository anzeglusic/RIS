import json
from pprint import pprint
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier

lib_of_files2 = {}
name = "/home/code8master/Desktop/wsROS/src/RIS/perception2/safe.json"
with open(name,"rt",encoding="utf-8") as f:
    lib_of_files2 = json.loads(f.read())

# print(lib_of_files2["list"])
print("numOfFiles")
print(f"\t{len(lib_of_files2)-1}\n")

lenCounter = {}
colorsList = []
for img in lib_of_files2["list"]:
    if len(img) in lenCounter:
        lenCounter[len(img)] += 1
    else:
        lenCounter[len(img)] = 1
    if len(img)==13:
        colorsList.append(img)
print("lenCounter:")
print(f"\t{lenCounter}\n")

colorCounter = {}
for img in colorsList:
    if img["color"] in colorCounter:
        colorCounter[img["color"]] += 1
    else:
        colorCounter[img["color"]] = 1
print("colorCounter:")
print(f"\t{colorCounter}\n")

counter = 0
permutations = {}
for i in range(0,4):
    for j in range(0,4):
        if (j==i):
            continue
        for k in range(0,4):
            if (k==i) or (k==j):
                continue
            counter += 1
            permutations[counter] = (i,j,k)
            # print(f"{counter}:\t{i} {j} {k}")

print("Permutations:")
pprint(permutations)

listOfPermutatedColors = []
for img in colorsList:
    for key in permutations:
        (i,j,k) = permutations[key]
        listOfPermutatedColors.append({  "color":img["color"]
                                        , "b0": img[f"b{i}"]
                                        , "g0": img[f"g{i}"]
                                        , "r0": img[f"r{i}"]

                                        , "b1": img[f"b{j}"]
                                        , "g1": img[f"g{j}"]
                                        , "r1": img[f"r{j}"]
                                        
                                        , "b2": img[f"b{k}"]
                                        , "g2": img[f"g{k}"]
                                        , "r2": img[f"r{k}"]
                                        })
print("\nNumber of permutated colors:")
print(f"\t{len(listOfPermutatedColors)}\n")




df = pandas.DataFrame(listOfPermutatedColors)
df.to_pickle("dfOfPermutatedColors.pkl")


print(df)

