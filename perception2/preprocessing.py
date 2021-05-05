import json
from pprint import pprint
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from skimage.color import rgb2hsv
import cv2

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


for i,img in enumerate(colorsList):
    rgb0 = np.array([[[img["r0"], img["g0"], img["b0"]]]]).astype(np.uint8)
    rgb1 = np.array([[[img["r1"], img["g1"], img["b1"]]]]).astype(np.uint8)
    rgb2 = np.array([[[img["r2"], img["g2"], img["b2"]]]]).astype(np.uint8)
    rgb3 = np.array([[[img["r3"], img["g3"], img["b3"]]]]).astype(np.uint8)
    # rgb0 = np.array([[[255,0,5]]]).astype(np.uint8)
    hsvColor0 = cv2.cvtColor(rgb0,cv2.COLOR_RGB2HSV)[0][0]
    hsvColor0[0] = int((float(hsvColor0[0])/180)*255)

    hsvColor1 = cv2.cvtColor(rgb1,cv2.COLOR_RGB2HSV)[0][0]
    hsvColor1[0] = int((float(hsvColor1[0])/180)*255)

    hsvColor2 = cv2.cvtColor(rgb2,cv2.COLOR_RGB2HSV)[0][0]
    hsvColor2[0] = int((float(hsvColor2[0])/180)*255)

    hsvColor3 = cv2.cvtColor(rgb3,cv2.COLOR_RGB2HSV)[0][0]
    hsvColor3[0] = int((float(hsvColor3[0])/180)*255)
    # print(rgb0[0][0],hsvColor)

    colorsList[i]["r0"] = hsvColor0[0]
    colorsList[i]["g0"] = hsvColor0[1]
    colorsList[i]["b0"] = hsvColor0[2]

    colorsList[i]["r1"] = hsvColor1[0]
    colorsList[i]["g1"] = hsvColor1[1]
    colorsList[i]["b1"] = hsvColor1[2]

    colorsList[i]["r2"] = hsvColor2[0]
    colorsList[i]["g2"] = hsvColor2[1]
    colorsList[i]["b2"] = hsvColor2[2]

    colorsList[i]["r3"] = hsvColor3[0]
    colorsList[i]["g3"] = hsvColor3[1]
    colorsList[i]["b3"] = hsvColor3[2]
    


colorDistributionCounter = {}
for img in colorsList:
    if img["color"] in colorDistributionCounter:
        colorDistributionCounter[img["color"]] += 1
    else:
        colorDistributionCounter[img["color"]] = 1
print("colorDistributionCounter:")
print(f"\t{colorDistributionCounter}\n")

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

print("-----")
df_np = df.to_numpy()
print(df_np)


Y = df_np[:,0]
X = df_np[:,1:]

print("\nY:")
print(Y)
print("\nX:")
print(X)

randomForest = RandomForestClassifier(n_estimators=100
                                    , criterion="gini"
                                    , max_depth=None
                                    , min_samples_split=2
                                    , min_samples_leaf=1
                                    , min_weight_fraction_leaf=0.0
                                    , max_features="auto"
                                    , max_leaf_nodes=None
                                    , min_impurity_decrease=0.0
                                    , bootstrap=True
                                    , oob_score=True
                                    , n_jobs=-1
                                    , random_state=None
                                    , verbose=0
                                    , warm_start=False
                                    , class_weight=None
                                    , ccp_alpha=0.0
                                    , max_samples=400
                                    )



print("-------------------------------------------------------------")
randomForest_fit = randomForest.fit(X,Y)
print("\nRandom Forest:")
print(f"\tscore: {randomForest_fit.oob_score_}\n")

print("-------------------------------------------------------------")

knn = KNeighborsClassifier(   n_neighbors=5
                            , weights="distance"
                            , algorithm="auto"
                            , n_jobs=-1
                            )
kf = KFold(n_splits=10, shuffle=True)
loopCounter = 0
overAllCounterCorrect = 0
overAllCounterFalse = 0
for train,test in kf.split(df_np):
    # print(df_np.shape)
    # print(train.shape)
    # print(test.shape)
    Y_kf = Y[train]
    X_kf = X[train,:]
    Y_kf_test = Y[test]
    X_kf_test = X[test,:]
    knn.fit(X_kf, Y_kf)
    prediction = knn.predict(X_kf_test)

    correctCounter = 0
    wrongCounter = 0
    colorDistribution = { "b":0
                        , "g":0
                        , "r":0
                        , "y":0
                        , "w":0
                        , "c":0
                        }
    for i in range(X_kf_test.shape[0]):
        if prediction[i] == Y_kf_test[i]:
            correctCounter += 1
            overAllCounterCorrect += 1
        else:
            wrongCounter += 1
            overAllCounterFalse += 1
        colorDistribution[prediction[i]] += 1
    print(f"\tKNN[split:{loopCounter}]:")
    print(f"\t\tcolorDistribution: {colorDistribution}")
    print(f"\t\tcorrectCounter:    {correctCounter}")
    print(f"\t\twrongCounter:      {wrongCounter}")
    loopCounter += 1

print(f"KNN:")
print(f"\toverAllCoutnerCorrect: {overAllCounterCorrect}")
print(f"\toverAllCoutnerFalse:   {overAllCounterFalse}")
print(f"\tscore:                 {overAllCounterCorrect/(overAllCounterCorrect+overAllCounterFalse)}")
print("-------------------------------------------------------------")

#----------------------------------------------------------------------------------------
def calc_rgb_distance_mask(point):
    p0 = point[0:3] 
    p1 = point[3:6]
    p2 = point[6:9]
    # print(f"p0: {p0}")
    # print(f"p1: {p1}")
    # print(f"p2: {p2}")

    distances = { "red":{}
                , "blue":{}
                , "green": {}
                , "black": {}
                , "white": {}
                , "yellow": {}
                }
    # red
    color = np.array([0,0,255])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["red"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["red"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["red"]["p2"] = dist

    #  blue
    color = np.array([255,0,0])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p2"] = dist
    
    # cyan --> blue
    color = np.array([255,255,0])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p0"] = min(dist,distances["blue"]["p0"])
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p1"] = min(dist,distances["blue"]["p1"])
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["blue"]["p2"] = min(dist,distances["blue"]["p2"])

    # green
    color = np.array([0,255,0])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["green"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["green"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["green"]["p2"] = dist

    # black
    color = np.array([0,0,0])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["black"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["black"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["black"]["p2"] = dist

    # white
    color = np.array([255,255,255])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["white"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["white"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["white"]["p2"] = dist

    # yellow
    color = np.array([0,255,255])
    calculatedPoint = p0-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["yellow"]["p0"] = dist
    
    calculatedPoint = p1-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["yellow"]["p1"] = dist
    
    calculatedPoint = p2-color
    dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
    distances["yellow"]["p2"] = dist
    

    bestScore = np.array([1,1,1])*np.sqrt(255**2 + 255**2 + 255**2)
    color = ["nothing","nothing","nothing"]
    for colorKey in distances:
        score0 = distances[colorKey]["p0"]
        if score0 < bestScore[0]:
            bestScore[0] = score0
            color[0] = colorKey
        
        score1 = distances[colorKey]["p1"]
        if score1 < bestScore[1]:
            bestScore[1] = score1
            color[1] = colorKey
        
        score2 = distances[colorKey]["p2"]
        if score2 < bestScore[2]:
            bestScore[2] = score2
            color[2] = colorKey
    
    counter = np.array([0,0,0])
    counter[0] = np.sum([color[0]==color[0],color[1]==color[0],color[2]==color[0]])
    counter[1] = np.sum([color[0]==color[1],color[1]==color[1],color[2]==color[1]])
    counter[2] = np.sum([color[0]==color[2],color[1]==color[2],color[2]==color[2]])
    # print(counter)
    if np.sum(counter)==3:
        col = color[np.argmin(bestScore)]
    else:
        col = color[np.argmax(counter)]
    
    if col == "black":
        return "c"
    if col == "white":
        return "w"
    if col == "blue":
        return "b"
    if col == "yellow":
        return "y"
    if col == "red":
        return "r"
    if col == "green":
        return "g"
#----------------------------------------------------------------------------------------

correctCounter = 0
wrongCounter = 0
colorDistribution = { "b":0
                    , "g":0
                    , "r":0
                    , "y":0
                    , "w":0
                    , "c":0
                    }
for i in range(X.shape[0]):
    foundColor = calc_rgb_distance_mask(X[i,:])
    if foundColor == Y[i]:
        correctCounter += 1
    else:
        wrongCounter += 1
    colorDistribution[foundColor] += 1
print("RGB-cube-distnace:")
print(f"\tcolorDistribution: {colorDistribution}")
print(f"\tcorrectCounter:    {correctCounter}")
print(f"\twrongCounter:      {wrongCounter}")