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
from JonasToolBox import JonasToolBox
from graph_widget import MatplotFigure
import matplotlib.pyplot as pt
import pandas as pd
import math
from plotCreation import plotCreator
Window.maximize()
jtb = JonasToolBox()
plotCreator = plotCreator()

class ConfigurationScreen(Screen):
    def loadMetaFile(self, path, filename):
        jtb.setupConfig(filename[0])

        self.ids.informationbox.clear_widgets()
        self.manager.get_screen("selection").reloadSelectionScreen()
        informationList = BoxLayout(orientation='vertical')
        titleBox = BoxLayout(orientation='vertical')
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text='Experiment Title', font_size=20, size_hint_x=0.3))
        titleBox.add_widget(AnchorBox)
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text=jtb.config_design["META"]["experiment_title"], size_hint_x=0.4))
        titleBox.add_widget(AnchorBox)

        informationList.add_widget(titleBox)
        passationBox = BoxLayout(orientation='vertical')
        AnchorBox = AnchorLayout(anchor_x='left')
        AnchorBox.add_widget(Label(text='Recording sites', font_size=20, size_hint_x=0.3))
        passationBox.add_widget(AnchorBox)
        for site in jtb.config_design["META"]["passation_site"]:
            AnchorBox = AnchorLayout(anchor_x='left')
            lab = Label(text="-" + site, size_hint_x=1, halign='left')
            lab.bind(size=lab.setter('text_size'))
            AnchorBox.add_widget(lab)
            passationBox.add_widget(AnchorBox)

        informationList.add_widget(passationBox)
        self.ids.informationbox.add_widget(informationList)


class SelectionScreen(Screen):
    lastcallback = None
    lastcallbackPopupFilter = None
    lastcallbackPopupGroup = None
    confirmbutton=Button(text='confirm')
    def reloadSelectionScreen(self):
        self.ids.databox.clear_widgets()
        print(jtb.profiling_summary.to_string())
        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Mean: ' + str(jtb.profiling_summary.loc['mean', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Min: ' + str(jtb.profiling_summary.loc['min', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Age Max: ' + str(jtb.profiling_summary.loc['max', ('demographics', 'age')]), size_hint_x=1,
                    halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        AnchorBox = AnchorLayout(anchor_x='left')
        lab = Label(text='Most Popular Gender: ' + str(
            jtb.profiling_summary.loc['top', ('demographics', 'gender')]) + ' ' + str(
            jtb.profiling_summary.loc['freq', ('demographics', 'gender')]) + "%", size_hint_x=1, halign='left')
        lab.bind(size=lab.setter('text_size'))
        AnchorBox.add_widget(lab)
        self.ids.databox.add_widget(AnchorBox)

        dropdown = DropDown()
        for categorie in  list(jtb.config_criterias["PROFILING"].keys()):
            btn = Button(text=categorie, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.ids.profiling_categorie.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: self.dropdownitem(self.ids.profiling_categorie, x))
        self.reloadGraph()

    def reloadGraph(self):
        pt.clf()
        sel = jtb.current_profiling_df.droplevel(axis='columns', level='profiling_categories')
        if  jtb.current_group!=None:
            finalsel = sel.pivot(columns=jtb.current_group[1])[self.ids.profiling_item.text]
            pt.hist(finalsel, rwidth=0.8, label=finalsel.columns)
            pt.legend()
        else:
            finalsel=sel[self.ids.profiling_item.text]
            pt.hist(finalsel, rwidth=0.8)
        # jtb.current_profiling_df.plot(x=(self.ids.profiling_categorie.text, self.ids.profiling_item.text),y=('TPI_scores','engagement'),kind='bar')
        graph = MatplotFigure()
        graph.figure = pt.gcf()
        graph.axes = pt.gca()
        graph.regenerateAxesLimits()
        graph.fast_draw = False
        graph.touch_mode = 'zoombox'
        self.ids.graphbox.clear_widgets()
        self.ids.graphbox.add_widget(graph)

    def changeitem(self, button, newtext):
        setattr(button, 'text', newtext)
        self.reloadSelectionScreen()

    def dropdownitem(self, button, newtext):
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in jtb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        if self.lastcallback != None:
            self.ids.profiling_item.unbind(on_release=self.lastcallback)
        self.ids.profiling_item.bind(on_release=dropdown.open)
        self.lastcallback = dropdown.open
        dropdown.bind(on_select=lambda instance, x: self.changeitem(self.ids.profiling_item, x))
        self.reloadGraph()

    def addFilterDropDown(self, button, newtext):
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in jtb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.changeSelectCriteriaFilter(self.FiltrerPopup.ids.criteria_item_button, newtext,
                                  list(jtb.config_criterias["PROFILING"][newtext].keys())[0])
        if self.lastcallbackPopupFilter != None:
            self.FiltrerPopup.ids.criteria_item_button.unbind(on_release=self.lastcallbackPopupFilter)
        self.FiltrerPopup.ids.criteria_item_button.bind(on_release=dropdown.open)
        self.lastcallbackPopupFilter = dropdown.open
        dropdown.bind(
            on_select=lambda instance, x: self.changeSelectCriteriaFilter(self.FiltrerPopup.ids.criteria_item_button, newtext,
                                                                    x))
    def ChangeGroupDropDown(self,button,newtext):
        setattr(button, 'text', newtext)
        dropdown = DropDown()
        for item in jtb.config_criterias["PROFILING"][newtext].keys():
            btn = Button(text=item, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.changeSelectCriteriaGroup(self.GroupPopup.ids.criteria_item_button, newtext,
                                  list(jtb.config_criterias["PROFILING"][newtext].keys())[0])
        if self.lastcallbackPopupGroup != None:
            self.GroupPopup.ids.criteria_item_button.unbind(on_release=self.lastcallbackPopupGroup)
        self.GroupPopup.ids.criteria_item_button.bind(on_release=dropdown.open)
        self.lastcallbackPopupFilter = dropdown.open
        dropdown.bind(
            on_select=lambda instance, x: self.changeSelectCriteriaGroup(self.GroupPopup.ids.criteria_item_button, newtext,x))

    def changeSelectCriteriaGroup(self, button, categorie, newtext):
        setattr(button, 'text', newtext)
        self.GroupPopup.ids.filterbox.clear_widgets()
        self.GroupPopup.ids.popupbuttons.remove_widget(self.confirmbutton)
        self.confirmbutton = Button(text='Confirm')
        if jtb.config_criterias["PROFILING"][categorie][newtext]['type'] == "numeric_range":
           print("numeric group")
        elif jtb.config_criterias["PROFILING"][categorie][newtext]['type'] == "string_list":
            self.confirmbutton.bind(on_release=lambda btn: self.changeCurrentGroup(newtext, categorie))
        self.GroupPopup.ids.popupbuttons.add_widget(self.confirmbutton)

    def changeSelectCriteriaFilter(self, button, categorie, newtext):
        setattr(button, 'text', newtext)
        jtb.reloadConfig()
        self.FiltrerPopup.ids.filterbox.clear_widgets()
        self.FiltrerPopup.ids.popupbuttons.remove_widget(self.confirmbutton)
        self.confirmbutton=Button(text='Confirm')
        self.ids.removefilterbutton.disabled = False
        if jtb.config_criterias["PROFILING"][categorie][newtext]['type'] == "numeric_range":
            fromtospace = Factory.FromToBox()
            fromdropdown = DropDown()
            for number in range(math.ceil(jtb.config_criterias["PROFILING"][categorie][newtext]['from']),
                                math.ceil(jtb.config_criterias["PROFILING"][categorie][newtext]['to'])):
                btn = Button(text=str(number), size_hint_y=None, height=44)
                btn.bind(on_release=lambda btn: fromdropdown.select(btn.text))
                fromdropdown.add_widget(btn)

            fromtospace.ids.frombutton.bind(on_release=fromdropdown.open)
            fromdropdown.bind(
                on_select=lambda instance, x: setattr(fromtospace.ids.frombutton, 'text', x)
            )
            todropdown = DropDown()
            for number in range(math.ceil(jtb.config_criterias["PROFILING"][categorie][newtext]['from']),
                                math.ceil(jtb.config_criterias["PROFILING"][categorie][newtext]['to'])):
                btn = Button(text=str(number), size_hint_y=None, height=44)
                btn.bind(on_release=lambda btn: todropdown.select(btn.text))
                todropdown.add_widget(btn)

            fromtospace.ids.tobutton.bind(on_release=todropdown.open)
            todropdown.bind(
                on_select=lambda instance, x: setattr(fromtospace.ids.tobutton, 'text', x))
            self.FiltrerPopup.ids.filterbox.add_widget(fromtospace)
            self.confirmbutton.bind(
                on_release=lambda btn: self.addNumericFilter(fromtospace, newtext, categorie))

        elif jtb.config_criterias["PROFILING"][categorie][newtext]['type'] == "string_list":
            checkboxlist = BoxLayout(orientation='vertical')
            self.selectedvalues = list(map(str,jtb.config_criterias["PROFILING"][categorie][newtext]['list']))
            for value in jtb.config_criterias["PROFILING"][categorie][newtext]['list']:
                row = Factory.CheckBoxListRow()
                row.ids.mycheckbox.bind(active=self.changecheckboxvalue)
                row.ids.mylabel.text = str(value)
                checkboxlist.add_widget(row)
            self.FiltrerPopup.ids.filterbox.add_widget(checkboxlist)
            self.confirmbutton.bind(
                on_release=  lambda btn: self.addListFilter(newtext, categorie))
        self.FiltrerPopup.ids.popupbuttons.add_widget(self.confirmbutton)

    def addListFilter(self,criteria, categorie):
        if self.selectedvalues[0].isdecimal():
            self.selectedvalues=list(map(int,self.selectedvalues))
        if self.FiltrerPopup.ids.unionbutton.state == 'down':
            data_sel = jtb.profiling_df.loc[jtb.profiling_df[(categorie, criteria)].isin(self.selectedvalues)]
            df_union = pd.concat([data_sel, jtb.current_profiling_df]).drop_duplicates()
            jtb.updateProfilingDf(df_union)
            self.ids.filtersbox.add_widget(
                Label(text=criteria))

        else:
            data_sel = jtb.current_profiling_df.loc[jtb.profiling_df[(categorie, criteria)].isin(self.selectedvalues)]
            jtb.updateProfilingDf(data_sel)
            self.ids.filtersbox.add_widget(
                Label(text=criteria))

        self.reloadSelectionScreen()
        self.FiltrerPopup.dismiss()

    def addNumericFilter(self, space, criteria, categorie):
        fromvalue = int(space.ids.frombutton.text)
        tovalue = int(space.ids.tobutton.text)
        if self.FiltrerPopup.ids.unionbutton.state == 'down':
            data_sel = jtb.profiling_df.loc[jtb.profiling_df[(categorie, criteria)] >= fromvalue]
            data_sel = data_sel.loc[data_sel[(categorie, criteria)] <= tovalue]
            df_union = pd.concat([data_sel, jtb.current_profiling_df]).drop_duplicates()
            jtb.updateProfilingDf(df_union)
            self.ids.filtersbox.add_widget(
                Label(text=space.ids.frombutton.text + '<=' + criteria + '<=' + space.ids.tobutton.text))

        else:
            data_sel = jtb.current_profiling_df.loc[jtb.profiling_df[(categorie, criteria)] > fromvalue]
            jtb.updateProfilingDf(data_sel.loc[data_sel[(categorie, criteria)] < tovalue])
            self.ids.filtersbox.add_widget(
                Label(text=space.ids.frombutton.text + '<=' + criteria + '<=' + space.ids.tobutton.text))

        self.reloadSelectionScreen()
        self.FiltrerPopup.dismiss()

    def changeCurrentGroup(self,item, categorie):
        jtb.changeGroup(categorie,item)
        self.ids.removegroupbutton.disabled=False
        self.ids.addgroupbutton.text='Remove Group'
        self.ids.groupsbox.clear_widgets()
        self.ids.groupsbox.add_widget(Label(text=categorie+' '+item))
        self.reloadSelectionScreen()
        self.GroupPopup.dismiss()

    def resetGroup(self):
        jtb.resetGroup()
        self.ids.removegroupbutton.disabled=True
        self.ids.addgroupbutton.text = 'Add Group'
        self.ids.groupsbox.clear_widgets()
        self.reloadSelectionScreen()
        self.GroupPopup.dismiss()

    def changecheckboxvalue(self, checkbox, value):
        if value and checkbox.parent.ids.mylabel.text not in self.selectedvalues:
            self.selectedvalues.append(checkbox.parent.ids.mylabel.text)
        else:
            self.selectedvalues.remove(checkbox.parent.ids.mylabel.text)

    def loadPopupFilter(self):
        self.FiltrerPopup = Factory.CustomPopup()
        dropdown = DropDown()
        tgb = ToggleButton(text='union')
        self.FiltrerPopup.ids.unionbutton = tgb
        self.FiltrerPopup.ids.buttonbox.add_widget(tgb)

        for criteria in jtb.config_criterias["PROFILING"].keys():
            btn = Button(text=criteria, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.FiltrerPopup.ids.criteria_categorie_button.bind(on_release=dropdown.open)
        dropdown.bind(
            on_select=lambda instance, x: self.addFilterDropDown(self.FiltrerPopup.ids.criteria_categorie_button, x))

        self.FiltrerPopup.open()

    def loadPopupGroup(self):
        self.GroupPopup = Factory.CustomPopup()
        dropdown = DropDown()
        for criteria in jtb.config_criterias["PROFILING"].keys():
            btn = Button(text=criteria, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)

        self.GroupPopup.ids.criteria_categorie_button.bind(on_release=dropdown.open)
        dropdown.bind(
            on_select=lambda instance, x: self.ChangeGroupDropDown(self.GroupPopup.ids.criteria_categorie_button, x))

        self.GroupPopup.open()

    def removeFilters(self):
        jtb.resetFilter()
        self.ids.removefilterbutton.disabled=True
        self.ids.filtersbox.clear_widgets()
        self.reloadSelectionScreen()

    def loadVizualisationScreen(self):
        jtb.updateMetaFileMask()
        self.manager.get_screen("visualization").reloadVizualisationGraph()
        self.ids.vizualisationshortcut.disabled=False
        self.manager.current = 'visualization'
        self.manager.transition.direction = 'left'



class VisualizationScreen(Screen):
    def reloadVizualisationGraph(self):
        self.ids.vizualisationgraph.clear_widgets()
        profilingplot=plotCreator.createProfilingPlot(jtb.current_profiling_df[("demographics","age")],jtb.current_profiling_df[("TPI_scores","engagement")])
        self.profilinggraph = MatplotFigure()
        self.profilinggraph.figure = profilingplot.gcf()
        self.profilinggraph.axes = profilingplot.gca()
        self.profilinggraph.regenerateAxesLimits()
        self.profilinggraph.fast_draw = False
        self.profilinggraph.touch_mode = 'zoombox'
        self.ids.vizualisationgraph.add_widget(self.profilinggraph)
        '''
        actionplot=plotCreator.createAggregatedPlot(jtb.data,"normal")
        self.actiongraph=MatplotFigure()
        self.actiongraph.figure= actionplot.gcf()
        self.actiongraph.axes=actionplot.gca()
        self.actiongraph.regenerateAxesLimits()
        self.actiongraph.fast_draw=False
        self.actiongraph.touch_mode='zoombox'
        self.ids.vizualisationgraph.add_widget(self.actiongraph)
        '''
class MyScreenManager(ScreenManager):
    pass


root_widget = Builder.load_string('''
MyScreenManager:
    ConfigurationScreen:
    SelectionScreen:
    VisualizationScreen:

<CustomPopup@Popup>:
    auto_dismiss: False
    size_hint:[0.7,0.7]
    BoxLayout:
        orientation:'vertical'
        AnchorLayout:
            anchor_x:'left'
            size_hint_y:0.2
            BoxLayout:
                size_hint_x:0.7
                orientation:'horizontal'
                id:buttonbox
                Label:
                    text:'Add the Following Criteria :'
                DropDownButton:
                    id:criteria_categorie_button
                    text:
                        'Select the categorie'
                DropDownButton:
                    id:criteria_item_button
                    text:
                        'Select the criteria to add'
                
        BoxLayout:
            id:filterbox
            
        BoxLayout:
            orientation:'horizontal'
            size_hint_y:0.1
            id:popupbuttons
            Button:
                text: 'Abandon'
                on_release: root.dismiss()

    
        
        
<FromToBox@BoxLayout>:
    BoxLayout:
        orientation:'horizontal'
        Label:
            text:'From'
        
        AnchorLayout:
            anchor_x:'center'
            Button:
                size_hint:[0.7,0.2]
                id:frombutton
                text:'select'
        Label:
            text:'To'
        AnchorLayout:
            anchor_x:'center'
            Button:
                size_hint:[0.7,0.2]
                text:'select'
                id:tobutton

<CheckBoxListRow@BoxLayout>:
    orientation:'horizontal'
    CheckBox:
        active:
            'down'
        id: mycheckbox
            
    Label:
        text:
            'default'
        id: mylabel
               
<DropDownButton@Button>:   
    canvas.after:
        Color:
            rgba: 0.5,0.5,0.5,0.5
        Triangle:
            points:(self.pos[0]+self.width-self.height/2,self.pos[1],self.pos[0]+self.width,self.pos[1],self.pos[0]+self.width,self.pos[1]+self.height/2)  
<NullButton@Button>:
    background_color: 1, 1, 1, 0
    
<DisabledRoundButton@NullButton>:
    canvas.before:
        Color:
            rgba: 0.2, 0.2, 0.2, 1
        RoundedRectangle:
            size: [self.height,self.height]
            pos: self.pos
            radius: [100]
                
<ActiveRoundButton@NullButton>:
    canvas.before:
        Color:
            rgba: 0.6, 0.6, 0.6, 1
        RoundedRectangle:
            size: [self.height,self.height]
            pos: self.pos
            radius: [100]
                
<CurrentRoundButton@NullButton>:
    canvas.before:
        Color:
            rgba: 0.95, 0.95, 0.95, 1
        RoundedRectangle:
            size: [self.height,self.height]
            pos: self.pos
            radius: [100]
 
<ExcludedBoxLayout@BoxLayout>:
    canvas.before:
        Line:
            width: 0.5
            rectangle: self.x, self.y+1, self.width-4, self.height+4
    
    
<Title@Label>:
    
    font_size:20      
    
<LeftAnchorLayout@AnchorLayout>:
    anchor_x:'left'            
<ConfigurationScreen>:
    name: 'configuration'
    BoxLayout:
        orientation: 'vertical' 
        BoxLayout:
            size_hint_y:0.15
            orientation: 'horizontal'
            Label: 
                size_hint_x:0.4
                text: 'Configuration'
            AnchorLayout:
                BoxLayout:
                    size_hint:[0.9,0.7]
                    orientation: 'horizontal'
                    CurrentRoundButton:
                        text:'Configuration'
                        on_release:app.root.current='configuration'
                    DisabledRoundButton:
                        text:'Selection'
                    DisabledRoundButton:
                        text:'Visualization'

        BoxLayout:
            orientation: 'horizontal'
            BoxLayout:
                orientation: 'vertical'
                ExcludedBoxLayout:
                    orientation: 'vertical'
                    LeftAnchorLayout:
                        size_hint_y: 0.05
                        Title:
                            text: 'Select the metafile'
                    FileChooserListView:
                        id: filechooser
                        dirselect: True
                    AnchorLayout:
                        size_hint_y: 0.1
                        anchor_x:'right'
                        Button:
                            text: 'Load'
                            size_hint_x:0.4
                            on_release: 
                                root.loadMetaFile(filechooser.path, filechooser.selection)
                AnchorLayout:
                    size_hint_y:0.4
                    Button:
                        size_hint: [0.4,0.3]
                        text:'Confirm'
                        on_release:
                            app.root.current='selection'
                            app.root.transition.direction='left'
            ExcludedBoxLayout:
                orientation:'vertical'
                
                LeftAnchorLayout:
                    size_hint_y: 0.05
                    Title:
                        text: 'Informations'
                BoxLayout:
                    id : informationbox
                    orientation:'vertical'
                    Label:
                        text:'None'
<SelectionScreen>:
    name: 'selection' 
    BoxLayout:
        orientation: 'vertical' 
        BoxLayout:
            size_hint_y:0.15
            orientation: 'horizontal'
            Label: 
                size_hint_x:0.4
                text: 'Selection'
            AnchorLayout:
                BoxLayout:
                    size_hint:[0.9,0.7]
                    orientation: 'horizontal'
                    ActiveRoundButton:
                        text:'Configuration'
                        on_release:
                            app.root.current='configuration'
                            app.root.transition.direction='right'
                    CurrentRoundButton:
                        text:'Selection'
                        on_release:app.root.current='selection'
                    DisabledRoundButton:
                        on_release:
                            app.root.current='visualization' 
                            app.root.transition.direction='left'
                        disabled:True
                        id:vizualisationshortcut
                        text:'Visualization'

        BoxLayout:
            orientation: 'horizontal'
            ExcludedBoxLayout:
                orientation: 'vertical'
                BoxLayout:
                    orientation: 'vertical'
                    LeftAnchorLayout:
                        size_hint_y: 0.1
                        Title: 
                            text: 'Filters'
                    BoxLayout:
                        id:filtersbox
                        orientation:'vertical'
                    BoxLayout:
                        size_hint_y: 0.3
                        orientation:'horizontal'
                        Button:
                            size_hint_x:0.5
                            text: 'Add Filter'
                            on_release: root.loadPopupFilter()
                        Button:
                            text: 'Remove Filters'
                            id:removefilterbutton
                            on_release: root.removeFilters()
                            disabled: True
                BoxLayout:
                    orientation: 'vertical'
                    LeftAnchorLayout:
                        size_hint_y: 0.1
                        Title:
                            text: 'Groups'
                    BoxLayout:
                        id:groupsbox
                        
                    BoxLayout:
                        size_hint_y: 0.3
                        orientation:'horizontal'
                        Button:
                            size_hint_x:0.5
                            text: 'Add Group'
                            id:addgroupbutton
                            on_release: root.loadPopupGroup()
                            
                        Button:
                            text: 'Remove Group'
                            id:removegroupbutton
                            on_release:root.resetGroup()
                            disabled: True

            BoxLayout:
                orientation: 'vertical'
                ExcludedBoxLayout:
                    orientation: 'vertical'
                    LeftAnchorLayout:
                        size_hint_y: 0.1
                        Title:
                            text: 'Data summary'
                    BoxLayout:
                        id:databox
                        orientation:'vertical'
                AnchorLayout:
                    size_hint_y:0.3
                    Button:
                        size_hint: [0.7,0.4]
                        text:'Confirm'
                        on_release:
                            root.loadVizualisationScreen()
            ExcludedBoxLayout:
                orientation: 'vertical'
                size_hint_x:2.5
                LeftAnchorLayout:
                    size_hint_y: 0.1
                    Title:
                        text: 'Graph'
                BoxLayout:
                    orientation:'vertical'
                    BoxLayout:
                        orientation:'horizontal'
                        BoxLayout:
                            id:graphbox
                    AnchorLayout:
                        anchor_x:'right'
                        size_hint_y:0.1
                        BoxLayout:
                            size_hint_x:0.5
                            orientation:'horizontal'
                            DropDownButton:
                                text:'demographics'
                                id:profiling_categorie 
                            DropDownButton:
                                text:'age'      
                                id:profiling_item 
 
                
<VisualizationScreen>:
    name: 'visualization'
    BoxLayout:
        orientation: 'vertical' 
        BoxLayout:
            size_hint_y:0.15
            orientation: 'horizontal'
            Label: 
                size_hint_x:0.4
                text: 'Visualization'
            AnchorLayout:
                BoxLayout:
                    size_hint:[0.9,0.7]
                    orientation: 'horizontal'
                    ActiveRoundButton:
                        text:'Configuration'
                        on_release:
                            app.root.current='configuration'
                            app.root.transition.direction='right'
                    ActiveRoundButton:
                        text:'Selection'
                        on_release:
                            app.root.current='selection'
                            app.root.transition.direction='right'
                    CurrentRoundButton:
                        text:'Visualization'
                        on_release:app.root.current='visualization'
        BoxLayout:
            orientation:'vertical'
            AnchorLayout:
                size_hint_y:0.1
                anchor_x:'left'
                BoxLayout:
                    size_hint_x:0.8
                    orientation:'horizontal'
                   
                    Button:
                        text:'Action'
                    
                    Button:
                        text:'Movement'
                  
                    Button:
                        text:'Metrics'
                   
                    Button:
                        text:'Survey data'
            BoxLayout:
                orientation:'horizontal'
                ExcludedBoxLayout:
                    id: vizualisationgraph
                    Label:
                        text: 'Graph here'
                ExcludedBoxLayout:
                    size_hint_x:0.3
                    orientation:'vertical'
''')


class XRVizApp(App):
    def build(self):
        return root_widget


XRVizApp().run()
