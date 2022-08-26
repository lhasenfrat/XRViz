from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.factory import Factory
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.dropdown import DropDown
from AnalysisToolBox import AnalysisToolBox
from graph_widget import MatplotFigure
import matplotlib.pyplot as pt
import pandas as pd
import math
from plotCreation import plotCreator

Window.maximize()
atb = AnalysisToolBox()
plotCreator = plotCreator()


class ConfigurationScreen(Screen):
    '''
    First screen of the Kivy app.
    The user can load the experiment from here and then proceed to the selection screen
    '''

    def loadMetaFile(self, path, filename):
        '''
        Loads metadatas from the experiment into the information box

        :param path: unused
        :param filename: file name given by the filechooser widget
        '''
        atb.setupConfig(filename[0])
        self.ids.informationbox.clear_widgets()
        self.manager.get_screen("selection").reloadSelectionScreen()

        informationList = BoxLayout(orientation='vertical')
        titleBox = BoxLayout(orientation='vertical')
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text='Experiment Title', font_size=20, size_hint_x=0.3))
        titleBox.add_widget(AnchorBox)
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text=atb.config_design["META"]["experiment_title"], size_hint_x=0.4))
        titleBox.add_widget(AnchorBox)

        informationList.add_widget(titleBox)
        passationBox = BoxLayout(orientation='vertical')
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text='Recording sites', font_size=20, size_hint_x=0.3))
        passationBox.add_widget(AnchorBox)
        for site in atb.config_design["META"]["passation_site"]:
            AnchorBox = AnchorLayout(anchor_x='left')
            lab = Label(text="-" + site, size_hint_x=1, halign='left')
            lab.bind(size=lab.setter('text_size'))
            AnchorBox.add_widget(lab)
            passationBox.add_widget(AnchorBox)

        informationList.add_widget(passationBox)
        self.ids.informationbox.add_widget(informationList)


class SelectionScreen(Screen):
    '''
    Second screen of the Kivy app.
    The user can select the participants to analyse, filtering with profiling data.
    Current selection is shown with data summary and with the graph display.
    The data summary needs to be customize for each experiment.
    TODO: make it more generic
    '''
    lastcallback = None
    lastcallbackPopupFilter = None
    lastcallbackPopupGroup = None
    confirmbutton = Button(text='confirm')

    def reloadSelectionScreen(self):
        '''
        Called when the experiment is selected.
        Fill the selection screen with experiment data, such as the data summary or the graph.
        Data summary is not generic and needs to be redone for each experiment to make it more relatable.
        '''
        self.ids.databox.clear_widgets()
        print(atb.profiling_summary.to_string())
        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Number of participants: ' + str(
            atb.current_profiling_df.nunique()[('experimental', 'subject_id')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(
            text='Number of condition: ' + str(atb.profiling_summary.loc['unique', ('experimental', 'condition')]),
            size_hint_x=1,
            halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(
            text='Number of passation sites: ' + str(
                atb.profiling_summary.loc['unique', ('experimental', 'passation_site')]),
            size_hint_x=1,
            halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        genderlist = list(set(atb.current_profiling_df[('demographics', 'gender')]))
        AnchorBox = AnchorLayout(anchor_x='left', size_hint_y=max(1, len(genderlist)))
        Genderlist = "\n".join(genderlist)
        lab = Label(text='Gender list : ' + Genderlist, size_hint_x=1, halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Mean: ' + str(atb.profiling_summary.loc['mean', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Min: ' + str(atb.profiling_summary.loc['min', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Max: ' + str(atb.profiling_summary.loc['max', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        for fpt_items in list(atb.config_criterias["PROFILING"]["FPT_scores"].keys()):
            AnchorBox = AnchorLayout(anchor_x='left')
            lab = Label(text=fpt_items + ' Mean: ' + "{:0.2f}".format(
                atb.profiling_summary.loc['mean', ('FPT_scores', fpt_items)]),
                        size_hint_x=1,
                        halign='left')
            lab.bind(size=lab.setter('text_size'))
            AnchorBox.add_widget(lab)
            self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Engagement Mean: ' + "{:0.2f}".format(
            atb.profiling_summary.loc['mean', ('TPI_scores', 'engagement')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        dropdown = DropDown()
        for categorie in list(atb.config_criterias["PROFILING"].keys()):
            btn = Button(text=categorie, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.ids.profiling_categorie.bind(on_release=dropdown.open)
        dropdown.bind(
            on_select=lambda instance, x: self.dropdownitem(self.ids.profiling_categorie, x, self.ids.profiling_item))
        self.reloadGraph()

    def reloadGraph(self):
        '''
        Create the plot based on the axis selected and displays it.
        If a group is selected then display the data based on groups.
        TODO: implement the groups system for numeric values
        '''
        pt.clf()
        sel = atb.current_profiling_df.droplevel(axis='columns', level='profiling_categories')
        if atb.group_type == "object":
            finalsel = sel.pivot(columns=atb.current_group[1])[self.ids.profiling_item.text]
            pt.hist(finalsel, rwidth=0.8, label=finalsel.columns)
            pt.legend()
        else:
            finalsel = sel[self.ids.profiling_item.text]
            pt.hist(finalsel, rwidth=0.8)
        pt.xlabel(self.ids.profiling_categorie.text + " : " + self.ids.profiling_item.text)
        pt.ylabel("Records")
        graph = MatplotFigure()
        graph.figure = pt.gcf()
        graph.axes = pt.gca()
        graph.regenerateAxesLimits()
        graph.fast_draw = False
        graph.touch_mode = 'zoombox'
        self.ids.graphbox.clear_widgets()
        self.ids.graphbox.add_widget(graph)

    def changeCurrentGroup(self, item, categorie):
        '''
        Add a group and reload the screen to updaet the graph
        :param item:new group item name
        :param categorie: new group categorie name
        '''
        atb.changeGroup(categorie, item)
        self.ids.removegroupbutton.disabled = False
        self.ids.addgroupbutton.text = 'Remove Group'
        self.ids.groupsbox.clear_widgets()
        self.ids.groupsbox.add_widget(Label(text=categorie + ' ' + item))
        self.reloadSelectionScreen()
        self.GroupPopup.dismiss()

    def resetGroup(self):
        '''
        remove the current group
        '''
        atb.resetGroup()
        self.ids.removegroupbutton.disabled = True
        self.ids.addgroupbutton.text = 'Add Group'
        self.ids.groupsbox.clear_widgets()
        self.reloadSelectionScreen()
        self.GroupPopup.dismiss()

    def changecheckboxvalue(self, checkbox, value):
        '''
        Change selected values for a filter based on strings
        :param checkbox: checkbox that as been activated
        :param value: current value of the checkbox (down or normal)
        '''
        if value and checkbox.parent.ids.mylabel.text not in self.selectedvalues:
            self.selectedvalues.append(checkbox.parent.ids.mylabel.text)
        else:
            self.selectedvalues.remove(checkbox.parent.ids.mylabel.text)

    def loadPopupFilter(self):
        '''
        Create the popup for adding a filter
        '''

        self.FiltrerPopup = Factory.CustomPopup()
        dropdown = DropDown()
        tgb = ToggleButton(text='union')
        self.FiltrerPopup.ids.unionbutton = tgb
        self.FiltrerPopup.ids.buttonbox.add_widget(tgb)

        for criteria in atb.config_criterias["PROFILING"].keys():
            btn = Button(text=criteria, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.FiltrerPopup.ids.criteria_categorie_button.bind(on_release=dropdown.open)
        dropdown.bind(
            on_select=lambda instance, x: self.addFilterDropDown(self.FiltrerPopup.ids.criteria_categorie_button, x))

        self.FiltrerPopup.open()

    def loadPopupGroup(self):
        '''
        Create the popup for adding a group
        '''
        self.GroupPopup = Factory.CustomPopup()
        dropdown = DropDown()
        for criteria in atb.config_criterias["PROFILING"].keys():
            btn = Button(text=criteria, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.GroupPopup.ids.criteria_categorie_button.bind(on_release=dropdown.open)
        dropdown.bind(
            on_select=lambda instance, x: self.ChangeGroupDropDown(self.GroupPopup.ids.criteria_categorie_button, x))

        self.GroupPopup.open()

    def removeFilters(self):
        '''
        remove all filters
        '''
        atb.resetFilter()
        self.ids.removefilterbutton.disabled = True
        self.ids.filtersbox.clear_widgets()
        self.reloadSelectionScreen()

    def changeitem(self, button, newtext):
        '''
        called when item for the plot axis is changed.
        Reload the screen to update the graph.
        :param button: button to change name.
        :param newtext: new item name.
        '''
        setattr(button, 'text', newtext)
        self.reloadSelectionScreen()

    def dropdownitem(self, button, newtext, nextbutton):
        '''
        Called when categorie for the plot axis is changed.
        :param button: categorie button.
        :param newtext: new categorie name.
        :param nextbutton: item button.
        '''
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in atb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        if self.lastcallback != None:
            nextbutton.unbind(on_release=self.lastcallback)
        nextbutton.bind(on_release=dropdown.open)
        self.lastcallback = dropdown.open
        dropdown.bind(on_select=lambda instance, x: self.changeitem(nextbutton, x))
        self.reloadGraph()

    def addFilterDropDown(self, button, newtext):
        '''
        Called when categorie for the filter popup is changed.
        :param button: categorie button.
        :param newtext: new categorie name.
        '''
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in atb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.changeSelectCriteriaFilter(self.FiltrerPopup.ids.criteria_item_button, newtext,
                                        list(atb.config_criterias["PROFILING"][newtext].keys())[0])
        if self.lastcallbackPopupFilter != None:
            self.FiltrerPopup.ids.criteria_item_button.unbind(on_release=self.lastcallbackPopupFilter)
        self.FiltrerPopup.ids.criteria_item_button.bind(on_release=dropdown.open)
        self.lastcallbackPopupFilter = dropdown.open
        dropdown.bind(
            on_select=lambda instance, x: self.changeSelectCriteriaFilter(self.FiltrerPopup.ids.criteria_item_button,
                                                                          newtext,
                                                                          x))

    def ChangeGroupDropDown(self, button, newtext):
        '''
        Called when categorie for the group popup is changed.
        :param button: categorie button.
        :param newtext: new categorie name.
        '''
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in atb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.changeSelectCriteriaGroup(self.GroupPopup.ids.criteria_item_button, newtext,
                                       list(atb.config_criterias["PROFILING"][newtext].keys())[0])
        if self.lastcallbackPopupGroup != None:
            self.GroupPopup.ids.criteria_item_button.unbind(on_release=self.lastcallbackPopupGroup)
        self.GroupPopup.ids.criteria_item_button.bind(on_release=dropdown.open)
        self.lastcallbackPopupGroup = dropdown.open
        dropdown.bind(
            on_select=lambda instance, x: self.changeSelectCriteriaGroup(self.GroupPopup.ids.criteria_item_button,
                                                                         newtext, x))

    def changeSelectCriteriaGroup(self, button, categorie, newtext):
        '''
        Called when item for the filter popup is changed.
        :param button: categorie button.
        :param categorie: categorie name.
        :param newtext: new item name.
        '''
        setattr(button, 'text', newtext)
        self.GroupPopup.ids.filterbox.clear_widgets()
        self.GroupPopup.ids.popupbuttons.remove_widget(self.confirmbutton)
        self.confirmbutton = Button(text='Confirm')
        if atb.config_criterias["PROFILING"][categorie][newtext]['type'] == "numeric_range":
            self.confirmbutton.bind(on_release=lambda btn: self.changeCurrentGroup(newtext, categorie))
        elif atb.config_criterias["PROFILING"][categorie][newtext]['type'] == "string_list":
            self.confirmbutton.bind(on_release=lambda btn: self.changeCurrentGroup(newtext, categorie))
        self.GroupPopup.ids.popupbuttons.add_widget(self.confirmbutton)

    def changeSelectCriteriaFilter(self, button, categorie, newtext):
        '''
        Called when item for the group popup is changed.
        If the new item is numeric the filter is based on a range, if it's a list then add the filter is based on selection of value.
        :param button: categorie button.
        :param categorie: categorie name.
        :param newtext: new item name.
        '''
        setattr(button, 'text', newtext)
        atb.reloadConfig()
        self.FiltrerPopup.ids.filterbox.clear_widgets()
        self.FiltrerPopup.ids.popupbuttons.remove_widget(self.confirmbutton)
        self.confirmbutton = Button(text='Confirm')
        self.ids.removefilterbutton.disabled = False
        if atb.config_criterias["PROFILING"][categorie][newtext]['type'] == "numeric_range":
            fromtospace = Factory.FromToBox()
            fromdropdown = DropDown()
            for number in range(math.ceil(atb.config_criterias["PROFILING"][categorie][newtext]['from']),
                                math.ceil(atb.config_criterias["PROFILING"][categorie][newtext]['to'])):
                btn = Button(text=str(number), size_hint_y=None, height=44)
                btn.bind(on_release=lambda btn: fromdropdown.select(btn.text))
                fromdropdown.add_widget(btn)

            fromtospace.ids.frombutton.bind(on_release=fromdropdown.open)
            fromdropdown.bind(
                on_select=lambda instance, x: setattr(fromtospace.ids.frombutton, 'text', x)
            )
            todropdown = DropDown()
            for number in range(math.ceil(atb.config_criterias["PROFILING"][categorie][newtext]['from']),
                                math.ceil(atb.config_criterias["PROFILING"][categorie][newtext]['to'])):
                btn = Button(text=str(number), size_hint_y=None, height=44)
                btn.bind(on_release=lambda btn: todropdown.select(btn.text))
                todropdown.add_widget(btn)

            fromtospace.ids.tobutton.bind(on_release=todropdown.open)
            todropdown.bind(
                on_select=lambda instance, x: setattr(fromtospace.ids.tobutton, 'text', x))
            self.FiltrerPopup.ids.filterbox.add_widget(fromtospace)
            self.confirmbutton.bind(
                on_release=lambda btn: self.addNumericFilter(fromtospace, newtext, categorie))

        elif atb.config_criterias["PROFILING"][categorie][newtext]['type'] == "string_list":
            checkboxlist = BoxLayout(orientation='vertical')
            self.selectedvalues = list(map(str, atb.config_criterias["PROFILING"][categorie][newtext]['list']))
            for value in atb.config_criterias["PROFILING"][categorie][newtext]['list']:
                row = Factory.CheckBoxListRow()
                row.ids.mycheckbox.bind(active=self.changecheckboxvalue)
                row.ids.mylabel.text = str(value)
                checkboxlist.add_widget(row)
            self.FiltrerPopup.ids.filterbox.add_widget(checkboxlist)
            self.confirmbutton.bind(
                on_release=lambda btn: self.addListFilter(newtext, categorie))
        self.FiltrerPopup.ids.popupbuttons.add_widget(self.confirmbutton)

    def addListFilter(self, criteria, categorie):
        '''
        Add a filter based on selected values.
        :param criteria: item name the filter affects.
        :param categorie: categorie name the filter affects.
        '''
        if self.selectedvalues[0].isdecimal():
            self.selectedvalues = list(map(int, self.selectedvalues))
        if self.FiltrerPopup.ids.unionbutton.state == 'down':
            data_sel = atb.profiling_df.loc[atb.profiling_df[(categorie, criteria)].isin(self.selectedvalues)]
            df_union = pd.concat([data_sel, atb.current_profiling_df]).drop_duplicates()
            atb.updateProfilingDf(df_union)
            self.ids.filtersbox.add_widget(
                Label(text=criteria))

        else:
            data_sel = atb.current_profiling_df.loc[atb.profiling_df[(categorie, criteria)].isin(self.selectedvalues)]
            atb.updateProfilingDf(data_sel)
            self.ids.filtersbox.add_widget(
                Label(text=criteria))

        self.reloadSelectionScreen()
        self.FiltrerPopup.dismiss()

    def addNumericFilter(self, space, criteria, categorie):
        '''
        Add a filter based on a range.
        :param space: Box where range bounds are stored.
        :param criteria: item name the filter affects.
        :param categorie: categorie name the filter affects.
        '''
        fromvalue = int(space.ids.frombutton.text)
        tovalue = int(space.ids.tobutton.text)
        if self.FiltrerPopup.ids.unionbutton.state == 'down':
            data_sel = atb.profiling_df.loc[atb.profiling_df[(categorie, criteria)] >= fromvalue]
            data_sel = data_sel.loc[data_sel[(categorie, criteria)] <= tovalue]
            df_union = pd.concat([data_sel, atb.current_profiling_df]).drop_duplicates()
            atb.updateProfilingDf(df_union)
            self.ids.filtersbox.add_widget(
                Label(text=space.ids.frombutton.text + '<=' + criteria + '<=' + space.ids.tobutton.text))

        else:
            data_sel = atb.current_profiling_df.loc[atb.profiling_df[(categorie, criteria)] > fromvalue]
            atb.updateProfilingDf(data_sel.loc[data_sel[(categorie, criteria)] < tovalue])
            self.ids.filtersbox.add_widget(
                Label(text=space.ids.frombutton.text + '<=' + criteria + '<=' + space.ids.tobutton.text))

        self.reloadSelectionScreen()
        self.FiltrerPopup.dismiss()

    def loadVizualisationScreen(self):
        '''
        Load action and object data from selected participants,
        create a graph and change the screen to the vizualisation screen.
        '''
        atb.updateMetaFileMask()
        self.manager.get_screen("visualization").reloadMetricsGraph()
        self.ids.vizualisationshortcut.disabled = False
        self.manager.current = 'visualization'
        self.manager.transition.direction = 'left'


class VisualizationScreen(Screen):
    '''
    Third screen of the Kivy app.
    The user can switch between two graphs : an Action graph and a Metrics graph.
    Action graph is slow to compute and show every action happening for every record selected.
    Metrics graph allow you to take two metrics(like surveys items or generated metrics like experiment durations).
    Groups are displayed with colors in the metrics graph and are not implemented in the action graph.
    '''
    lastcallback_x = None
    lastcallback_y = None
    profilinggraph = None
    actiongraph = None

    def reloadActionsGraph(self):
        '''
        Call plotCreator to create the action graph and displays it.
        '''
        if self.actiongraph == None:
            # the parameter "normal" refer to information quantity displayed by the action plot,
            # can be set to "reduced" to sum up information in basic ligns
            actionplot = plotCreator.createAggregatedPlot(atb.data, "normal")
            self.actiongraph = MatplotFigure()
            self.actiongraph.figure = actionplot.gcf()
            self.actiongraph.axes = actionplot.gca()
            self.actiongraph.regenerateAxesLimits()
            self.actiongraph.fast_draw = False
            self.actiongraph.touch_mode = 'zoombox'
        self.ids.VizualisationBox.clear_widgets()
        self.ids.VizualisationBox.add_widget(self.actiongraph)

    def reloadMetricsGraph(self):
        '''
        Load metrics into the dropdown buttons.
        '''
        self.graph = Factory.MetricsGraph()
        dropdown_x = DropDown()
        for categorie in list(atb.config_criterias["PROFILING"].keys()):
            btn = Button(text=categorie, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown_x.select(btn.text))
            dropdown_x.add_widget(btn)

        dropdown_y = DropDown()
        for categorie in list(atb.config_criterias["PROFILING"].keys()):
            btn = Button(text=categorie, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown_y.select(btn.text))
            dropdown_y.add_widget(btn)

        self.graph.ids.profiling_categorie_x.bind(on_release=dropdown_x.open)
        self.graph.ids.profiling_categorie_y.bind(on_release=dropdown_y.open)

        dropdown_x.bind(
            on_select=lambda instance, x: self.dropdownitem(self.graph.ids.profiling_categorie_x, x,
                                                            self.graph.ids.profiling_item_x, 'x'))
        dropdown_y.bind(
            on_select=lambda instance, x: self.dropdownitem(self.graph.ids.profiling_categorie_y, x,
                                                            self.graph.ids.profiling_item_y, 'y'))
        self.ids.VizualisationBox.clear_widgets()
        self.ids.VizualisationBox.add_widget(self.graph)
        self.reloadGraph()

    def reloadGraph(self):
        '''
        Call plotCreator to create the metrics graph and displays it.
        '''
        profilingplot = plotCreator.createMetricsPlot(
            (self.graph.ids.profiling_categorie_x.text, self.graph.ids.profiling_item_x.text),
            (self.graph.ids.profiling_categorie_y.text, self.graph.ids.profiling_item_y.text),
            atb.current_profiling_df, atb.group_type, atb.current_group)
        profilingplot.xlabel(self.graph.ids.profiling_categorie_x.text + " : " + self.graph.ids.profiling_item_x.text)
        profilingplot.ylabel(self.graph.ids.profiling_categorie_y.text + " : " + self.graph.ids.profiling_item_y.text)
        self.profilinggraph = MatplotFigure()
        self.profilinggraph.figure = profilingplot.gcf()
        self.profilinggraph.axes = profilingplot.gca()
        self.profilinggraph.regenerateAxesLimits()
        self.profilinggraph.fast_draw = False
        self.profilinggraph.touch_mode = 'zoombox'
        self.graph.ids.vizualisationgraph.clear_widgets()
        self.graph.ids.vizualisationgraph.add_widget(self.profilinggraph)

    def dropdownitem(self, button, newtext, nextbutton, axis):
        '''
        called when categorie for an axis is changed.
        update the item dropdown button.
        :param button: button to change name.
        :param newtext: new categorie name.
        :param nextbutton: item button.
        :param axis: x or y axis. used to make the method reusable.
        '''
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in atb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        if axis == 'x':
            if self.lastcallback_x != None:
                nextbutton.unbind(on_release=self.lastcallback_x)
            self.lastcallback_x = dropdown.open
        else:
            if self.lastcallback_y != None:
                nextbutton.unbind(on_release=self.lastcallback_y)
            self.lastcallback_y = dropdown.open

        nextbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: self.changeitem(nextbutton, x))

    def changeitem(self, button, newtext):
        '''
        called when item for an axis is changed.
        Reload the screen to update the graph.
        :param button: button to change name.
        :param newtext: new item name.
        '''
        setattr(button, 'text', newtext)
        self.reloadGraph()


class MyScreenManager(ScreenManager):
    pass




class XRVizApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(ConfigurationScreen(name='menu'))
        sm.add_widget(SelectionScreen(name='selection'))
        sm.add_widget(VisualizationScreen(name='visualization'))

        return sm

XRVizApp().run()
