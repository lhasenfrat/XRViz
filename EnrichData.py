import os
import csv
import pandas as pd

teleportdistance = 1
turnangle = 40


def distance2D(vector1, vector2):
    return ((vector1[0] - vector2[0]) ** 2 + (vector1[1] - vector2[1]) ** 2) ** (0.5)


def distance(vector1, vector2):
    return ((vector1[0] - vector2[0]) ** 2 + (vector1[1] - vector2[1]) ** 2 + (vector1[2] - vector2[2]) ** 2) ** (0.5)


def angle(vector1, vector2):
    return abs(vector2[2] - vector1[2])


def actionIdToObjectId(actionId):
    if actionId > 0:
        return actionId - 1
    return 0


x = -1


def getNextInt():
    global x
    x += 1
    return x


def resetInt():
    global x
    x = -1


spaces = {}
spaces["hall"] = [[-4.458, 0.692], [-1.495, -1.251]]
spaces["corridor"] = [[-3.57, -1.26], [-2.359, -7.117]]
spaces["bedroom 1"] = [[-8.1, -1.27], [-3.61, -4.57]]
spaces["bedroom 2"] = [[-2.32, -4.72], [1.53, -9.43]]
spaces["bathroom"] = [[-6.65, -4.72], [-3.61, -8.03]]
spaces["livingroom"] = [[-2.3, -1.26], [5.56, -4.54]]
spaces["kitchen"] = [[1.64, -4.54], [5.56, -8.01]]
spaces["terrace"] = [[5.61, 0.48], [8.14, -6.46]]

for root, dirs, files in os.walk("Enregistrements", topdown=False):

    for fileName in files:
        try:
            if "objectsData" not in fileName:
                continue
            date, data, scene = fileName.split("_")

            fileActionFormat = open(root + '\\' + date + "_actionsFormat_" + scene, 'w')
            actionFormatWriter = csv.writer(fileActionFormat, lineterminator='\n')
            actionFormatWriter.writerow(
                ["ActionId", "ActionName", "NbParam", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6",
                 "Param7",
                 "Param8", "Param9"])
            actionFormatWriter.writerow(
                ["1", "Teleport", "6", "StartPosition X", "StartPosition Y", "StartPosition Z", "EndPosition X",
                 "EndPosition Y", "EndPosition Z"])
            actionFormatWriter.writerow(
                ["2", "Turn", "2", "Activated", "Direction"])
            actionFormatWriter.writerow(
                ["3", "Hover", "3", "Activated", "Interactable(objectId)", "distanceToInteractable"])

            actionFormatWriter.writerow(
                ["4", "Select", "3", "Activated", "Interactable(objectId)", "distanceToInteractable"])
            actionFormatWriter.writerow(["5", "EnterSpace", "1", "Space(objectId)"])
            newfile = open(root + '\\' + date + "_actionsData_" + scene, 'w')
            writer = csv.writer(newfile, lineterminator='\n')
            writer.writerow(
                ["timestamp", "ticksSince1970 (100ns)", "ObjectId", "ActionId", "Param1", "Param2", "Param3", "Param4",
                 "Param5", "Param6", "Param7", "Param8", "Param9"])

            objectsDataPanda = pd.read_csv(root + '\\' + fileName)
            try:
                objectsDataPanda["actionId"] = objectsDataPanda["actionId"].map(actionIdToObjectId)
                objectsDataPanda.rename(columns={"actionId": "objectId"}, inplace=True)
                objectsDataPanda.to_csv(root + '\\' + fileName, index=False)
            except:
                print("actionId already converted")
            resetInt()
            objectsFormatPanda = pd.read_csv(root + '\\' + date + "_objectsFormat_" + scene)
            objectsFormatPanda.rename(columns={"type": "objectId"}, inplace=True)
            objectsFormatPanda["objectId"] = objectsFormatPanda["objectId"].apply(lambda x: getNextInt())
            objectsFormatPanda.to_csv(root + '\\' + date + "_objectsFormat_" + scene, index=False)

            with open(root + '\\' + fileName, 'r') as objectsData:
                objectsDataContent = list(csv.reader(objectsData))
                objectsFormatContent = pd.read_csv(root + '\\' + date + "_objectsFormat_" + scene)
                objectsDic = {}
                i = 0
                currentRoom=-1
                newRoom=-1
                lastHMDposition = []
                lastHMDrotation = []
                currentLeftHandposition = []
                currentRightHandposition = []
                leftHandGrabedObject = -1
                rightHandGrabedObject = -1
                leftHandGrabedObjectLastMoved = []
                rightHandGrabedObjectLastMoved = []
                leftHandHoverObject = -1
                rightHandHoverObject = -1
                objectPositions = {}
                for objectName in objectsFormatContent["trackedData"]:
                    objectsDic[objectName] = i
                    if "Camera" in objectName:
                        cameraId = i
                    elif "LeftHand" in objectName:
                        leftHandId = i
                    elif "RightHand" in objectName:
                        rightHandId = i
                    i += 1
                for datarow in objectsDataContent[2:]:
                    currentposition = list(map(float, datarow[3:6]))
                    currentrotation = list(map(float, datarow[6:9]))

                    currentObject = int(datarow[2])
                    currentTimeStamp = float(datarow[0])
                    currentTick = float(datarow[1])
                    objectPositions[currentObject] = currentposition

                    ################################ MAIN CAMERA

                    if currentObject == cameraId:
                        if len(lastHMDposition) == 0:
                            lastHMDposition = currentposition
                        else:
                            if distance2D(currentposition, lastHMDposition) > teleportdistance:
                                writer.writerow(
                                    [currentTimeStamp, currentTick, currentObject, '1', lastHMDposition[0],
                                     lastHMDposition[1], lastHMDposition[2], currentposition[0], currentposition[1],
                                     currentposition[2]])

                            lastHMDposition = currentposition

                        if len(lastHMDrotation) == 0:
                            lastHMDrotation = currentrotation
                        else:
                            currentangle = angle(currentrotation, lastHMDrotation)
                            currentangle = min(currentangle, 360 - currentangle)

                            if currentangle > turnangle:
                                writer.writerow([currentTimeStamp, currentTick, currentObject, '2',1,"no idea"])

                            lastHMDrotation = currentrotation

                        if "House" in fileName :
                            for space in spaces:
                                if spaces[space][0][0]<currentposition[0]<spaces[space][1][0] and spaces[space][0][1]>currentposition[2]>spaces[space][1][1] :
                                    newRoom=space

                                    break
                            if newRoom!=currentRoom:
                                writer.writerow(
                                    [currentTimeStamp, currentTick, currentObject, '5', newRoom])


                            currentRoom=newRoom

                    ################################ LEFT HAND

                    elif currentObject == leftHandId:
                        currentLeftHandposition = currentposition

                        if leftHandHoverObject != -1:
                            distanceToInteractable = distance(
                                objectPositions[leftHandHoverObject], currentposition)
                            if distanceToInteractable > 0.3:
                                writer.writerow(
                                    [currentTimeStamp, currentTick, leftHandId, '3', '0', leftHandHoverObject,
                                     distanceToInteractable])
                                leftHandHoverObject = -1

                        if leftHandHoverObject == -1:
                            for object in objectPositions:
                                if object != currentObject:

                                    distanceToInteractable = distance(
                                        objectPositions[object], currentposition)

                                    if distanceToInteractable < 0.1:
                                        leftHandHoverObject = object
                                        writer.writerow(
                                            [currentTimeStamp, currentTick, leftHandId, '3', '1', leftHandHoverObject,
                                             distanceToInteractable])
                                        break

                        if len(leftHandGrabedObjectLastMoved) == 0:
                            continue
                        distanceToInteractable = distance(
                            leftHandGrabedObjectLastMoved, currentposition)
                        if leftHandGrabedObject != -1 and distanceToInteractable > 0.3:
                            writer.writerow(
                                [currentTimeStamp, currentTick, leftHandId, '4', '0', leftHandGrabedObject,
                                 distanceToInteractable])
                            leftHandGrabedObject = -1


                    ################################ RIGHT HAND

                    elif currentObject == rightHandId:
                        currentRightHandposition = currentposition

                        if rightHandHoverObject != -1:
                            distanceToInteractable = distance(
                                objectPositions[rightHandHoverObject], currentposition)
                            if distanceToInteractable > 0.3:
                                writer.writerow(
                                    [currentTimeStamp, currentTick, rightHandId, '3', '0', rightHandHoverObject,
                                     distanceToInteractable])
                                rightHandHoverObject = -1

                        if rightHandHoverObject == -1:
                            for object in objectPositions:
                                if object != currentObject:

                                    distanceToInteractable = distance(
                                        objectPositions[object], currentposition)

                                    if distanceToInteractable < 0.1:
                                        rightHandHoverObject = object
                                        writer.writerow(
                                            [currentTimeStamp, currentTick, rightHandId, '3', '1', rightHandHoverObject,
                                             distanceToInteractable])
                                        break

                        if len(rightHandGrabedObjectLastMoved) == 0:
                            continue
                        distanceToInteractable = distance(
                            rightHandGrabedObjectLastMoved, currentposition)
                        if rightHandGrabedObject != -1 and distanceToInteractable > 0.3:
                            writer.writerow(
                                [currentTimeStamp, currentTick, rightHandId, '4', '0', rightHandGrabedObject,
                                 distanceToInteractable])

                            rightHandGrabedObject = -1


                    ################################ TRACKED OBJECT

                    else:
                        if len(currentLeftHandposition) != 0:
                            distanceToLeftHand = distance(
                                currentLeftHandposition, currentposition)
                            if distanceToLeftHand < 0.1:
                                if leftHandGrabedObject != currentObject:
                                    writer.writerow(
                                        [currentTimeStamp, currentTick, leftHandId, '4', '1', currentObject,
                                         distanceToLeftHand])

                                    leftHandGrabedObject = currentObject
                                leftHandGrabedObjectLastMoved = currentposition
                        if len(currentRightHandposition) != 0:
                            distanceToRightHand = distance(
                                currentRightHandposition, currentposition)
                            if distanceToRightHand < 0.1:
                                if rightHandGrabedObject != currentObject:
                                    writer.writerow(
                                        [currentTimeStamp, currentTick, rightHandId, '4', '1', currentObject,
                                         distanceToRightHand])
                                    rightHandGrabedObject = currentObject
                                rightHandGrabedObjectLastMoved = currentposition

            print("file done")
        except Exception as e:
            print("error with file " + root + fileName + " : ", e)
