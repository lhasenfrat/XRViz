import pandas as pd
import os
import matplotlib.pyplot as pt

gaussianParameter = 10


class plotCreator():
    informationQuantity = "reduced"

    def blendColor(self, actionmap, position, color):
        actionmap[position][1] = [(actionmap[position][1][0] + color[0]) / 2,
                                  (actionmap[position][1][1] + color[1]) / 2,
                                  (actionmap[position][1][2] + color[2]) / 2]
        for i in range(gaussianParameter):
            if position - i > 0:
                actionmap[position - i][1] = [(actionmap[position - i][1][0] * i + color[0]) / (1 + i),
                                              (actionmap[position - i][1][1] * i + color[1]) / (1 + i),
                                              (actionmap[position - i][1][2] * i + color[2]) / (1 + i)]
            if position + i < len(actionmap):
                actionmap[position + i][1] = [(actionmap[position + i][1][0] * i + color[0]) / (1 + i),
                                              (actionmap[position + i][1][1] * i + color[1]) / (1 + i),
                                              (actionmap[position + i][1][2] * i + color[2]) / (1 + i)]

    def hexaFormat(self, number):
        hexaNumber = format(number, 'X')
        if len(hexaNumber) == 1:
            return "0" + hexaNumber
        return hexaNumber

    def createProfilingPlot(self,first_axis,second_axis):
        pt.cla()
        pt.clf()
        pt.hexbin(first_axis,second_axis,gridsize=20)
        return pt

    def createFigure(self, size):
        self.fig, self.ax = pt.subplots(size, squeeze=False,sharex=True,sharey=True)

    def createAggregatedPlot(self, data,informationQuantity):

        self.createFigure(len(data))
        x=0
        for i in range(len(data)):
            self.createPlot(data[i][1],data[i][4],data[i][5],data[i][3], x)
            x += 1
        return pt


    def createPlot(self,condition, data,actionformat,objectformat, x=0):
        objectgroups = data.groupby('ObjectId')
        if data.size==0:
            return pt
        size = (int(data.iloc[-1]["timestamp"]) + 1) * 10

        actionmap = []
        for i in range(size):
            actionmap.append([i / 10, [200, 200, 200]])
        for objectNumber, objectgroup in objectgroups:
            objectName = \
            (objectformat.loc[objectformat["objectId"] == objectNumber]["trackedData"]).to_string().split("    ")[1]
            actiongroups = objectgroup.groupby('ActionId')
            for actionNumber, actiongroup in actiongroups:
                actionName = \
                (actionformat.loc[actionformat["ActionId"] == actionNumber]["ActionName"]).to_string().split()[1]
                if actionName == "EnterSpace":
                    if (objectName.__contains__("Camera")):
                        for i in range(len(actiongroup["timestamp"]) - 1):
                            self.ax[x, 0].axvline(list(actiongroup["timestamp"])[i], color='b')
                            self.ax[x, 0].text(
                                (list(actiongroup["timestamp"])[i] + list(actiongroup["timestamp"])[i + 1]) / 2, -1,
                                list(actiongroup["Param1"])[i], fontsize=7)
                        self.ax[x, 0].axvline(list(actiongroup["timestamp"])[-1], color='b')
                        self.ax[x, 0].text(list(actiongroup["timestamp"])[-1] + 0.5, -1,
                                           list(actiongroup["Param1"])[-1], fontsize=7)

                else:
                    if actionName == "Hover" or actionName == "Select":
                        for i in range(len(actiongroup["timestamp"])):
                            if actionName == "Hover":
                                self.blendColor(actionmap, int(list(actiongroup["timestamp"])[i] * 10), [255, 255, 0])
                            if actionName == "Select":
                                self.blendColor(actionmap, int(list(actiongroup["timestamp"])[i] * 10), [255, 0, 0])
                            if self.informationQuantity == "normal":
                                self.ax[x, 0].plot(list(actiongroup["timestamp"])[i], list(actiongroup["ActionId"])[i],
                                                   color='y' if actionName=="Hover" else 'r', marker='o' if list(actiongroup["Param1"])[i] else 'x',
                                                   linestyle='')
                                self.ax[x, 0].text(list(actiongroup["timestamp"])[i] - 0.15,
                                                   list(actiongroup["ActionId"])[i] - 0.25,
                                                   int(list(actiongroup["Param2"])[i]), fontsize=7)
                    elif actionName == "Turn":
                        for i in range(len(actiongroup["timestamp"])):
                            self.blendColor(actionmap, int(list(actiongroup["timestamp"])[i] * 10), [0, 255, 0])
                            marker = ""
                            if list(actiongroup["Param2"])[i] == "left":
                                marker = "<"
                            elif list(actiongroup["Param2"])[i] == "right":
                                marker = ">"
                            else:
                                marker = "+"
                            if self.informationQuantity == "normal":
                                self.ax[x, 0].plot(list(actiongroup["timestamp"])[i], list(actiongroup["ActionId"])[i],
                                                   color='g',
                                                   marker=marker, linestyle='')
                    elif actionName == "Teleport":
                        for i in range(len(actiongroup["timestamp"])):
                            self.blendColor(actionmap, int(list(actiongroup["timestamp"])[i] * 10), [0, 0, 255])
                            newX = (float(list(actiongroup["Param1"])[i]) - float(list(actiongroup["Param4"])[i])) ** 2
                            newY = (float(list(actiongroup["Param2"])[i]) - float(list(actiongroup["Param5"])[i])) ** 2
                            newZ = (float(list(actiongroup["Param3"])[i]) - float(list(actiongroup["Param6"])[i])) ** 2
                            distance = (newX + newY + newZ) ** 0.5
                            if self.informationQuantity == "normal":
                                self.ax[x, 0].plot(list(actiongroup["timestamp"])[i], list(actiongroup["ActionId"])[i],
                                                   color='b',
                                                   marker='o', markersize=min(distance ** 1.5, 25), linestyle='')
                    else:
                        if self.informationQuantity == "normal":
                            self.ax[x, 0].plot(actiongroup["timestamp"], actiongroup["ActionId"], color='k',
                                               marker='o', linestyle='')
        for action in actionmap:
            newColor = "#" + self.hexaFormat(int(action[1][0])) + self.hexaFormat(int(action[1][1])) + self.hexaFormat(
                int(action[1][2]))
            self.ax[x, 0].plot(action[0], 0, newColor, alpha=0.5, marker='s', linestyle='', ms=5)
        if self.informationQuantity == "normal":
            self.ax[x, 0].set_yticks(list(actionformat["ActionId"])[:-1])
            self.ax[x, 0].set_yticklabels(list(actionformat["ActionName"])[:-1])
        else :
            self.ax[x,0].set_yticks([0])
            self.ax[x,0].set_yticklabels(["ActionMap"])
        self.ax[x, 0].set_title(condition,fontsize=7,pad=0)

        pt.xlabel("Time")
        return pt
