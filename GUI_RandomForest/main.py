# Libraries
import pandas as pd
import random
import pickle
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Div, Select
from bokeh.plotting import figure

#----------------------------------------------------------------------#
# Load files
features_info = pd.read_csv('GUI_RandomForest/static/features_CML02282022.csv')
s_media = np.array(pd.read_csv("GUI_RandomForest/static/mean_hourly_hypoglycemia_07012022.csv")['media'].values)

# Load model
filename = 'GUI_RandomForest/static/MERF_model/MODEL_MRF_RF_P_SKLEARN102_07022022.sav'
open_file = open(filename, "rb")
clf = pickle.load(open_file)
open_file.close()

#----------------------------------------------------------------------#
## Initialize variables

# Initial parameters
dict_variables = {"f_demographics_age":0,   "f_clinical_years_with_t1d":0,  "f_glucose_variability_lback_24h":0,    "f_lbgi_lback_24h":0,   "f_proxy_carbs_2h_prior_ref_24h":0, "f_glucose_roc_lback_30min":0,  "f_glucose_start":0,    "f_iob_start_norm_by_estTDIR":0,    "f_pa_activity_intensity_kcal_per_min":0,   "f_pa_activity_duration_min":0, "fv_pa_class_cardio":0, "fv_pa_class_strength":0,   "fv_pa_class_mixed":0,  "fv_pa_class_other":0,  "f_pa_activity_time_of_day_hour_sin":0, "f_pa_activity_time_of_day_hour_cos":0, "f_pa_time_since_prev_pa_event_hr":0}
dict_activities = {"fv_pa_class_cardio":"Cardio","fv_pa_class_strength":'Strenght',"fv_pa_class_mixed":"Mixed","fv_pa_class_other":"Other"}
dict_activities_inverse = {"Cardio":"fv_pa_class_cardio",'Strenght':"fv_pa_class_strength","Mixed":"fv_pa_class_mixed","Other":"fv_pa_class_other"}

physical_activities = ["fv_pa_class_cardio","fv_pa_class_strength","fv_pa_class_mixed","fv_pa_class_other"]

for i,f in enumerate(features_info['feature']):
    if  f ==  "fv_pa_class":
        pa = [features_info.loc[i,'initial_value']]
        pa_remaining = list(set(physical_activities)-set(pa))
        dict_variables[pa[0]] = 1
        for p in pa_remaining:
            dict_variables[p] = 0

    elif f =="f_pa_activity_time_of_day_hour":
        pa_initial = int(features_info.loc[i,'initial_value'])
        dict_variables["f_pa_activity_time_of_day_hour_cos"] = np.cos(2*np.pi*float(features_info.loc[i,'initial_value'])/24)
        dict_variables["f_pa_activity_time_of_day_hour_sin"] = np.sin(2*np.pi*float(features_info.loc[i,'initial_value'])/24)
    else:
        dict_variables[f] = float(features_info.loc[i,'initial_value'])

predict = pd.DataFrame(data = np.tile(np.array(list(dict_variables.values())),(25,1)), columns = dict_variables.keys() )
predict['t_following_pa'] = np.arange(0,25,1)

# Variables (x,y) for the first plot
t = np.arange(0,25,1) # Time
s = clf.predict(predict) # Hypoglycemia risk

s_red,t_red = [],[] # Hypoglycemia is lower than the mean hypoglycemia in the training set for a given hour.
s_green,t_green = [],[] # Hypoglycemia is highr than the mean hypoglycemia in the training set for a given hour.

for i in range(len(s)):
    if s[i]>=s_media[i]:
        s_red.append(s[i])
        t_red.append(t[i])
    else:
        s_green.append(s[i])
        t_green.append(t[i])

#----------------------------------------------------------------------#
# Start Visualization
source = ColumnDataSource(data=dict(x=t, y=s))

plot = figure(height=400, width=1600, title="",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[-0.5, 24.5],x_axis_label='Time',
              y_axis_label='Predicted probability of hypoglicemia')

# Define y label
f_hour = [str(i)+":00" for i in range(24)]
f_hour = f_hour+f_hour
f_hour_s = f_hour[pa_initial:pa_initial+24][::5]
hour = np.arange(0,24)[::5]
dict_hours = {}
for ind,h in enumerate(hour):
    dict_hours[int(h)] = f_hour_s[ind]

plot.xaxis.ticker = hour
plot.xaxis.major_label_overrides = dict_hours

# Plot
plot.scatter('x', 'y', source=source, size=10, line_alpha=1,marker="square",color='black')#,legend_label="Predicted")

source_red = ColumnDataSource(data=dict(x=t_red, y=s_red))
plot.scatter('x', 'y', marker="square",source=source_red, size=5, line_alpha=1,color='red')

source_green = ColumnDataSource(data=dict(x=t_green, y=s_green))
plot.scatter('x', 'y', source=source_green, size=5, line_alpha=1,marker="square",color='green')

# Define nightime
night_init = 22-pa_initial
night_end = 22-pa_initial+8
source_box1 = ColumnDataSource(data=dict(x=[night_init,night_end], y1=[min(s),min(s)],y2=[max(s),max(s)]))
source_box2 = ColumnDataSource(data=dict(x=[0,0], y1=[min(s),min(s)],y2=[max(s),max(s)]))

plot.varea(x='x',y1='y1',y2='y2',source=source_box1,fill_color='gray', fill_alpha=0.2)
plot.varea(x='x',y1='y1',y2='y2',source=source_box2,fill_color='gray', fill_alpha=0.2)

source_gray = ColumnDataSource(data=dict(x=[-10], y=[min(s)]))
plot.scatter('x','y',color='gray',line_alpha=1,size=20,marker="square",legend_label='Nighttime.22:00, 8h')

# Define widgets
sliders = []
text = []
for i in features_info.index:
    if features_info.loc[i,'feature']== "fv_pa_class":
        sliders.append(Select(title = features_info.loc[i,'show_name'],value=dict_activities[features_info.loc[i,'initial_value']],options=list(dict_activities.values()),name=features_info.loc[i,'feature'],width=300,height=50))
    else:
        sliders.append(Slider(title = features_info.loc[i,'show_name']+' '+features_info.loc[i,'units'], start = features_info.loc[i,'min_value'], end = features_info.loc[i,'max_value'], 
                             value=float(features_info.loc[i,'initial_value']),
                             step=features_info.loc[i,'delta_feature'], bar_color = (features_info.loc[i,'color_r'],features_info.loc[i,'color_g'],features_info.loc[i,'color_b']), 
                             orientation='horizontal',name=features_info.loc[i,'feature'],width=300,height=50))
    text.append(Div(text=features_info.loc[i,'description'],width=300,height=50))

buttons = []
feat_gr = features_info.set_index('group')
for g in feat_gr.index.unique():
    try:
        buttons.append(Div(text='<span style="background-color: rgb'+str((feat_gr.loc[g,'color_r'][0],feat_gr.loc[g,'color_g'][0],feat_gr.loc[g,'color_b'][0]))+'">&emsp;</span>&emsp;'+g, style={'font-size': '150%'},width=300,height=50))
    except:
        buttons.append(Div(text='<span style="background-color: rgb'+str((feat_gr.loc[g,'color_r'],feat_gr.loc[g,'color_g'],feat_gr.loc[g,'color_b']))+'">&emsp;</span>&emsp;'+g, style={'font-size': '150%'},width=300,height=50))
  
# Function to update any change in the widgets        
def update(attrname, old, new):
    for i,sl in enumerate(sliders):
        f = sl.name
        value = sl.value

        if  f ==  "fv_pa_class":
            pa = [dict_activities_inverse[value]]
            pa_remaining = list(set(physical_activities)-set(pa))
            dict_variables[pa[0]] = 1
            for p in pa_remaining:
                dict_variables[p] = 0

        elif f =="f_pa_activity_time_of_day_hour":
            pa_initial = int(value)
            dict_variables["f_pa_activity_time_of_day_hour_cos"] = np.cos(2*np.pi*float(value)/24)
            dict_variables["f_pa_activity_time_of_day_hour_sin"] = np.sin(2*np.pi*float(value)/24)


        else:
            dict_variables[f] = float(value)

    f_hour = [str(i)+":00" for i in range(24)]
    f_hour = f_hour+f_hour
    f_hour_s = f_hour[pa_initial:pa_initial+24][::5]
    hour = np.arange(0,24)[::5]
    dict_hours = {}
    for ind,h in enumerate(hour):
        dict_hours[int(h)] = f_hour_s[ind]
    
    predict = pd.DataFrame(data = np.tile(np.array(list(dict_variables.values())),(25,1)), columns = dict_variables.keys() )
    predict['t_following_pa'] = np.arange(0, 25, 1)
    #s = np.random.uniform(0,1,len(t))
    s = clf.predict(predict)

    source.data = dict(x=t, y=s)
    
    s_red,t_red = [],[]
    s_green,t_green = [],[]
    for i in range(len(s)):
        if s[i]>=s_media[i]:
            s_red.append(s[i])
            t_red.append(t[i])
        else:
            s_green.append(s[i])
            t_green.append(t[i])
    
    source_red.data = dict(x=t_red, y=s_red)
    source_green.data = dict(x=t_green, y=s_green)
    
    night_init_p = (22-pa_initial )%24
    night_end_p = (22-pa_initial +8)%24

    if pa_initial>=6 and pa_initial<=22:
        night_init = 22-pa_initial  
        night_end = 22-pa_initial+8
        source_box1.data = dict(x=[0,0], y1=[min(s),min(s)],y2=[max(s),max(s)])
        source_box2.data = dict(x=[night_init,night_end], y1=[min(s),min(s)],y2=[max(s),max(s)])
    
    else:
        night_init = night_init_p 
        night_end = 24
        source_box1.data = dict(x=[night_init,night_end], y1=[min(s),min(s)],y2=[max(s),max(s)])

        night_init = 0  
        night_end = night_end_p
        source_box2.data = dict(x=[night_init,night_end], y1=[min(s),min(s)],y2=[max(s),max(s)])
    
    source_gray.data = dict(x=[-10], y=[min(s)])

    plot.xaxis.major_label_overrides = dict_hours

# Check if any change in the widgets
for sl in sliders:
    sl.on_change('value',update)

# Set up layouts and add to document
text_ = " <figure style='width: 1500px;'> <img src='static/logo-OHSU-RGB-4C-POS.png' style='float:right;width:80px'> </figure> <h3><b>Physical activity and hypoglycemia risk in type 1 diabetes</b></h3><p>This random forest model provides an objective hourly hypoglycemia risk score for physical activity decision support.</p> <p>A <font color='#008000'>green marker</font> means that the predicted probability of hypoglycemia is lower than the mean hypoglycemia in the training set for a given hour.</p> <p>A <font color='#FF0000'>red marker</font> means that the predicted probability of hypoglycemia is higher or equal to the mean hypoglycemia in the training set for a given hour.</p>"

introduction = Div(text=text_,sizing_mode='fixed',width=1500,height=200,style={'font-size': '110%'})

inputs = row(sliders[0:5],sizing_mode='fixed',width=1500,height=50)
inputs_t = row(text[0:5],sizing_mode='fixed',width=1500,height=50)
inputs2 = row(sliders[5:10],sizing_mode='fixed',width=1500,height=50)
inputs2_t = row(text[5:10],sizing_mode='fixed',width=1500,height=50)

inputs3 = row(sliders[10:],sizing_mode='fixed',width=1500,height=50)
inputs3_t = row(text[10:],sizing_mode='fixed',width=1500,height=50)

inputs_buttons = row(buttons,sizing_mode='fixed',width=1500,height=50)

plots = row(plot,sizing_mode='fixed',width=1500,height=400)

curdoc().add_root(introduction)
curdoc().add_root(column([plots,inputs,inputs_t, inputs2,inputs2_t,inputs3,inputs3_t,inputs_buttons],width=1500))
curdoc().title = "PA - Hypoglycemia Risk"
