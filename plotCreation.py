import pandas as pd
import os
import matplotlib.pyplot as pt
from matplotlib import colors
import matplotlib.cm as cmx

gaussianParameter = 10


class plotCreator():
    informationQuantity = "reduced"

    def blendColor(self, actionmap, position, color):
        '''
        Blend colors when adding a for the action map (the line summing up actions for a record).
        :param actionmap: the actionmap for the record.
        :param position: position of the action added.
        :param color: color of the new action.
        '''
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
        '''
        Translate a number into the right hexadecimal format.
        :param number: number to translate.
        :return: translated number.
        '''
        hexaNumber = format(number, 'X')
        if len(hexaNumber) == 1:
            return "0" + hexaNumber
        return hexaNumber

    def createMetricsPlot(self,first_axis_name,second_axis_name,data,group_type,group_axis_name):
        '''
        Create the metrics plot for the visualization screen.
        :param first_axis_name: metric of the first axis.
        :param second_axis_name: metric of the second axis.
        :param data: profiling data.
        :param group_type: type of group, basically numeric or string.
        :param group_axis_name: metric used for the grouping.
        :return: plot.
        '''
        pt.cla()
        pt.clf()
        if group_type=="int64" or group_type=="float64":
            pt.scatter(data[first_axis_name],data[second_axis_name],c=data[group_axis_name])
            pt.colorbar(label=str(group_axis_name[0]+" : "+group_axis_name[1]))

        elif group_type=="object":
            grouplist = list(set(data[group_axis_name]))

            rainbow = pt.get_cmap('tab20b')
            cNorm = colors.Normalize(vmin=0, vmax=len(grouplist))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
            for i in range(len(grouplist)):
                pt.scatter(data[data[group_axis_name]==grouplist[i]][first_axis_name],data[data[group_axis_name]==grouplist[i]][second_axis_name],color=scalarMap.to_rgba(i), label=grouplist[i])
                pt.legend()
        else:
            pt.scatter(data[first_axis_name],data[second_axis_name])
        return pt

    def createFigure(self, size):
        '''
        setup of the plot.
        :param size: number of figure.
        '''
        self.fig, self.ax = pt.subplots(size, squeeze=False,sharex=True,sharey=True)

    def createAggregatedPlot(self, data,informationQuantity):
        '''
        create a action plot that is a list of multiple record based action plot.
        :param data: list of action data.
        :param informationQuantity: quantity of information displayed in the action graph. if setup to "reduced", only the action map is shown.
        :return: plot.
        '''
        self.informationQuantity = informationQuantity

        self.createFigure(len(data))
        x=0
        for i in data.keys():
            self.createPlot(data[i][1],data[i][4],data[i][5],data[i][3], x)
            x += 1
        return pt


    def createPlot(self,condition, data,actionformat,objectformat, x=0):
        '''
        Create a action plot.
        The action plot describe a record and every actions happening in it.
        :param condition: condition name for the title.
        :param data: action data.
        :param actionformat: action format : list of action present in action data.
        :param objectformat: object format : list of object present in object data.
        :param x: position of the figure.
        :return: plot.
        '''
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
                                (list(actiongroup["timestamp"])[i] + list(actiongroup["timestamp"])[i + 1]) / 2, 0,
                                list(actiongroup["Param1"])[i], fontsize=7)
                        self.ax[x, 0].axvline(list(actiongroup["timestamp"])[-1], color='b')
                        self.ax[x, 0].text(list(actiongroup["timestamp"])[-1] + 0.5, 0,
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
