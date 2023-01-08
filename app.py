import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Conv1D,LSTM,Bidirectional,GRU,MaxPooling1D,Flatten

app=Flask(__name__,template_folder='template')
model = tf.keras.models.load_model('my_h5_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    lst = ['toss_win',
     'venue_Barabati Stadium',
     'venue_Brabourne Stadium',
     'venue_Buffalo Park',
     'venue_De Beers Diamond Oval',
     'venue_Dr DY Patil Sports Academy',
     'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
     'venue_Dubai International Cricket Stadium',
     'venue_Eden Gardens',
     'venue_Feroz Shah Kotla',
     'venue_Green Park',
     'venue_Himachal Pradesh Cricket Association Stadium',
     'venue_Holkar Cricket Stadium',
     'venue_JSCA International Stadium Complex',
     'venue_Kingsmead',
     'venue_M Chinnaswamy Stadium',
     'venue_M.Chinnaswamy Stadium',
     'venue_MA Chidambaram Stadium, Chepauk',
     'venue_Maharashtra Cricket Association Stadium',
     'venue_Nehru Stadium',
     'venue_New Wanderers Stadium',
     'venue_Newlands',
     'venue_OUTsurance Oval',
     'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
     'venue_Punjab Cricket Association Stadium, Mohali',
     'venue_Rajiv Gandhi International Stadium, Uppal',
     'venue_Sardar Patel Stadium, Motera',
     'venue_Saurashtra Cricket Association Stadium',
     'venue_Sawai Mansingh Stadium',
     'venue_Shaheed Veer Narayan Singh International Stadium',
     'venue_Sharjah Cricket Stadium',
     'venue_Sheikh Zayed Stadium',
     "venue_St George's Park",
     'venue_Subrata Roy Sahara Stadium',
     'venue_SuperSport Park',
     'venue_Vidarbha Cricket Association Stadium, Jamtha',
     'venue_Wankhede Stadium',
     'team1_Chennai Super Kings',
     'team1_Deccan Chargers',
     'team1_Delhi Capitals',
     'team1_Delhi Daredevils',
     'team1_Gujarat Lions',
     'team1_Kings XI Punjab',
     'team1_Kochi Tuskers Kerala',
     'team1_Kolkata Knight Riders',
     'team1_Mumbai Indians',
     'team1_Pune Warriors',
     'team1_Rajasthan Royals',
     'team1_Rising Pune Supergiant',
     'team1_Rising Pune Supergiants',
     'team1_Royal Challengers Bangalore',
     'team1_Sunrisers Hyderabad',
     'team2_Chennai Super Kings',
     'team2_Deccan Chargers',
     'team2_Delhi Capitals',
     'team2_Delhi Daredevils',
     'team2_Gujarat Lions',
     'team2_Kings XI Punjab',
     'team2_Kochi Tuskers Kerala',
     'team2_Kolkata Knight Riders',
     'team2_Mumbai Indians',
     'team2_Pune Warriors',
     'team2_Rajasthan Royals',
     'team2_Rising Pune Supergiant',
     'team2_Rising Pune Supergiants',
     'team2_Royal Challengers Bangalore',
     'team2_Sunrisers Hyderabad',
     'toss_decision_bat',
     'toss_decision_field',
     'umpire1_A Deshmukh',
     'umpire1_A Nand Kishore',
     'umpire1_AK Chaudhary',
     'umpire1_AM Saheba',
     'umpire1_AV Jayaprakash',
     'umpire1_AY Dandekar',
     'umpire1_Aleem Dar',
     'umpire1_Asad Rauf',
     'umpire1_BF Bowden',
     'umpire1_BG Jerling',
     'umpire1_BNJ Oxenford',
     'umpire1_BR Doctrove',
     'umpire1_C Shamshuddin',
     'umpire1_CB Gaffaney',
     'umpire1_CK Nandan',
     'umpire1_DJ Harper',
     'umpire1_GAV Baxter',
     'umpire1_HDPK Dharmasena',
     'umpire1_IJ Gould',
     'umpire1_IL Howell',
     'umpire1_JD Cloete',
     'umpire1_K Bharatan',
     'umpire1_K Hariharan',
     'umpire1_K Srinath',
     'umpire1_KN Ananthapadmanabhan',
     'umpire1_M Erasmus',
     'umpire1_MR Benson',
     'umpire1_NJ Llong',
     'umpire1_Nitin Menon',
     'umpire1_PG Pathak',
     'umpire1_PR Reiffel',
     'umpire1_RE Koertzen',
     'umpire1_RJ Tucker',
     'umpire1_RK Illingworth',
     'umpire1_RM Deshpande',
     'umpire1_S Asnani',
     'umpire1_S Das',
     'umpire1_S Ravi',
     'umpire1_SD Fry',
     'umpire1_SJ Davis',
     'umpire1_SJA Taufel',
     'umpire1_SK Tarapore',
     'umpire1_SL Shastri',
     'umpire1_SS Hazare',
     'umpire1_UV Gandhe',
     'umpire1_VA Kulkarni',
     'umpire1_VK Sharma',
     'umpire1_YC Barde',
     'umpire2_A Deshmukh',
     'umpire2_A Nand Kishore',
     'umpire2_AK Chaudhary',
     'umpire2_AL Hill',
     'umpire2_AM Saheba',
     'umpire2_AV Jayaprakash',
     'umpire2_BG Jerling',
     'umpire2_BNJ Oxenford',
     'umpire2_BR Doctrove',
     'umpire2_C Shamshuddin',
     'umpire2_CB Gaffaney',
     'umpire2_CK Nandan',
     'umpire2_DJ Harper',
     'umpire2_GA Pratapkumar',
     'umpire2_HDPK Dharmasena',
     'umpire2_I Shivram',
     'umpire2_IJ Gould',
     'umpire2_IL Howell',
     'umpire2_JD Cloete',
     'umpire2_K Hariharan',
     'umpire2_K Srinath',
     'umpire2_K Srinivasan',
     'umpire2_M Erasmus',
     'umpire2_MR Benson',
     'umpire2_NJ Llong',
     'umpire2_Nitin Menon',
     'umpire2_PG Pathak',
     'umpire2_PR Reiffel',
     'umpire2_RB Tiffin',
     'umpire2_RE Koertzen',
     'umpire2_RJ Tucker',
     'umpire2_RK Illingworth',
     'umpire2_S Asnani',
     'umpire2_S Das',
     'umpire2_S Ravi',
     'umpire2_SD Fry',
     'umpire2_SD Ranade',
     'umpire2_SJ Davis',
     'umpire2_SJA Taufel',
     'umpire2_SK Tarapore',
     'umpire2_SL Shastri',
     'umpire2_SS Hazare',
     'umpire2_Subroto Das',
     'umpire2_TH Wijewardene',
     'umpire2_VA Kulkarni',
     'umpire2_VK Sharma',
     'umpire2_YC Barde']

    f = [x for x in request.form.values()]
    ft_tf = []
    ft_tf.append(int(f[0]))
    ft_tf.append('venue_'+f[1])
    ft_tf.append('team1_'+f[2])
    ft_tf.append('team2_'+f[3])
    ft_tf.append('toss_decision_'+f[4])
    ft_tf.append('umpire1_'+f[5])
    ft_tf.append('umpire2_'+f[6])
    final_feat = [ft_tf[0]]
    x = 1
    for i in lst[1:]:
        if x<len(f) and i==ft_tf[x]:
            final_feat.append(1)
            x = x+1
        else:
            final_feat.append(0)
    
    xx = np.array(final_feat).reshape(1,164)
    xx = np.array([xx])

    prediction = model.predict(xx)

    output = prediction[0][0]
    prd = None
    if output>0.5:
     prd = f[3]
    else:
     prd = f[2]
    return render_template('index.html', prediction_text='The Winner Team will be: '+prd)


if __name__ == "__main__":
    app.run(debug=True)