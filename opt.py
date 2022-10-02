import streamlit as st
import numpy as np
import datetime
import pandas as pd                        
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy_financial as npf
import warnings
import time
import math
warnings.filterwarnings("ignore") 

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>Loan Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: darkgrey;'>Optimize loan adjustment, calculation done for 4 years</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: darkgrey;'>&copy; Liting Aalto 2022 - All rights reserved</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    start = st.date_input(
    "start date of your loan",
    datetime.date(2022, 3, 30))
    end = st.date_input(
        "start date of loan protection",
        datetime.date(2022, 9, 6))
    m = int((pd.to_datetime(end) - pd.to_datetime(start))/np.timedelta64(1, 'M'))
with col2:
    total_loan = st.number_input('Total loan', value = 200000)
    margin = st.number_input('Loan margin', value = 0.005,format="%g")
    extra=st.number_input('Loan Protection Extra Margin', value = 0.0182,format="%g") 
    cap=st.number_input('Loan Protection Cap', value = 0.01,format="%g") 
    
with col3:
    euribor1 = st.number_input('Current euribor', value = 0,format="%g")
    euribor2 = st.number_input('Euribor 2023', value = 0.045,format="%g")
    euribor3 = st.number_input('Euribor 2024', value = 0.025,format="%g")
    euribor4 = st.number_input('Euribor 2025', value = 0.01, format="%g")
    
with st.expander("If your family is borrowing you money"):    
    col1, col2 = st.columns(2)
    with col1:
        help1 = st.number_input('Family help € 2023', value = 0)
        help2 = st.number_input('Family help €  2024', value = 0)
        help3 = st.number_input('Family help €  2025', value = 0)
    with col2:
        if help1 <= 0:
            bound1 = (0,1)
        else:
            bound1 = st.slider(f'help 2023 to normal loan', 0, int(help1), (0, int(help1)))
        if help2 <= 0:
            bound2 = (0,1)
        else:
            bound2 = st.slider(f'help 2024 to normal loan', 0, int(help2), (0, int(help2)))
        if help3 <= 0:
            bound3 = (0,1)
        else:
            bound3 = st.slider(f'help 2025 to normal loan', 0, int(help3), (0, int(help3)))
class Euribor:
    def __init__(self, margin, loan, t=299):
        self.margin = margin
        self.t = t
        self.loan = loan
        self.interest = 0
    def cal_normal(self, euribor, help):
        self.loan -= help
        ai = []
        al=[]
        rm = []
        if self.loan > 0:
            for i in np.arange(12):
                if npf.ppmt((self.margin + euribor)/12, i+1, self.t, self.loan) < 0 :
                    rm.append(self.loan)
                    self.interest +=  npf.ipmt((self.margin + euribor)/12, i+1, self.t, self.loan) * -1
                    self.loan += npf.ppmt((self.margin + euribor)/12, i+1, self.t, self.loan)
                    ai.append(npf.ipmt((self.margin + euribor)/12, i+1, self.t, self.loan) * -1)
                    al.append(npf.ppmt((self.margin + euribor)/12, i+1, self.t, self.loan) * -1)
                    self.t -= 1
                    
        return ai, al, rm
    def cal_protect(self, euribor, help, extra, cap):
        self.loan -= help
        ai = []
        al=[]
        rm = []
        if self.loan > 0:
            for i in np.arange(12):
                rm.append(self.loan)
                if npf.ppmt((self.margin + extra + min(euribor, cap))/12, i+ 1, self.t, self.loan) <0 :
                    self.interest += npf.ipmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan) * -1
                    self.loan += npf.ppmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan)
                    ai.append(npf.ipmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan) * -1)
                    al.append(npf.ppmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan) * -1)
                    self.t -= 1
        return ai, al, rm
    def cal_first(self, euribor, extra, cap, m ):
        ai = []
        al=[]
        rm = []
        for i in np.arange(m):
            rm.append(self.loan)
            self.interest += npf.ipmt((self.margin + euribor)/12, i + 1, self.t, self.loan) * -1
            self.loan += npf.ppmt((self.margin + euribor)/12,i + 1, self.t, self.loan)
            ai.append(npf.ipmt((self.margin + euribor)/12, i+1, self.t, self.loan) * -1)
            al.append(npf.ppmt((self.margin + euribor)/12, i+1, self.t, self.loan) * -1)
            self.t -= 1
        for i in np.arange(12-m):
            rm.append(self.loan)
            self.interest += npf.ipmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan)* -1
            self.loan += npf.ppmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan)
            ai.append(npf.ipmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan) * -1)
            al.append(npf.ppmt((self.margin + extra + min(euribor, cap))/12, i + 1, self.t, self.loan) * -1)
            self.t -= 1
        return ai, al, rm

def object_func(x,  help1, help2, help3, euribor2, euribor3, euribor4, extra, cap, margin, m):
    a=Euribor(margin, x[0])
    b = Euribor(margin, x[1])
    a.cal_normal(euribor1,0)
    b.cal_first(euribor1,extra, cap, m)
    
    a.cal_normal(euribor2, x[2])        
    b.cal_protect(euribor2, help1-x[2], extra, cap)

    a.cal_normal(euribor3, x[3])
    b.cal_protect(euribor3, help2-x[3], extra, cap)
    
    a.cal_normal(euribor4, x[4])
    b.cal_protect(euribor4, help3-x[4], extra, cap)
    print(a.loan, b.loan)
     
    return max(a.interest,0)+max(b.interest,0)

def print_func(x,  help1, help2, help3, euribor2, euribor3, euribor4, extra, cap, margin, m):
    dfa=pd.DataFrame()
    dfb=pd.DataFrame()
    a=Euribor(margin, x[0])
    b = Euribor(margin, x[1])
    ai, al, rm = a.cal_normal(euribor1,0)
    bi, bl, brm = b.cal_first(euribor1,extra, cap, m)
    if len(ai) > 0 and ai[0] >1:
        dfa['Interest Paid'] = ai
        dfa['Principal Paid'] = al
        dfa['Principal Remain'] = rm
        dfa['year'] = 'year 1'
        dfa['month'] = ['month '+str(i+1) for i in range(12)]
    if len(bi) > 0 and bi[0] >1:
        
        dfb['Interest Paid'] = bi
        dfb['Principal Paid'] = bl
        dfb['Principal Remain'] = brm
        dfb['year'] = 'year 1'
        dfb['month'] = ['month '+str(i+1) for i in range(12)]

    ai, al, rm = a.cal_normal(euribor2, x[2])  
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 2'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfa = dfa.append(df2)
    
    ai, al, rm = b.cal_protect(euribor2, help1-x[2], extra, cap)
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 2'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfb = dfb.append(df2)
    

    ai, al, rm =a.cal_normal(euribor3, x[3])
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 3'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfa = dfa.append(df2)
    ai, al, rm =b.cal_protect(euribor3, help2-x[3], extra, cap)
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 3'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfb = dfb.append(df2)
        
        
    ai, al, rm =a.cal_normal(euribor4, x[4])
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 4'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfa = dfa.append(df2)
    ai, al, rm =b.cal_protect(euribor4, help3-x[4], extra, cap)
    if len(ai) > 0 and ai[0] >1:
        df2 = pd.DataFrame([ai, al, rm]).T
        df2.columns=['Interest Paid','Principal Paid', 'Principal Remain']
        df2['year'] = 'year 4'
        df2['month'] = ['month '+str(i+1) for i in range(12)]
        dfb = dfb.append(df2)
     
    return dfa, dfb

def const(x):
    return x[0]+x[1]-total_loan

if st.button('optimize your loan'):
    
    cons=({'type':'eq','fun': const})
    result = minimize(object_func, [0,0,0,0,0],
                      args=(help1, help2, help3, euribor2, euribor3, euribor4, extra, cap, margin,m),
                      constraints = cons,
                      bounds = [(0,total_loan),(0,total_loan),bound1, bound2, bound3])
    if help1 + help2 + help3 > 0:

        st.metric(label="Total interests", value=int(result.fun))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Normal loan", value=int(round(result.x[0],0)))
            st.metric(label="Loan protection", value=int(round(result.x[1],0)))
        with col2:
            st.metric(label="help 2023", value=int(round(result.x[2],0)))
            st.metric(label="help 2023", value=int(help1-round(result.x[2],0)))
        with col3:
            st.metric(label="help 2024", value=int(round(result.x[3],0)))
            st.metric(label="help 2024", value=int(help2-round(result.x[3],0)))
        with col4:
            st.metric(label="help 2025", value=int(round(result.x[4],0)))
            st.metric(label="help 2025", value=int(help3-round(result.x[4],0)))
    else:
        st.metric(label="Total interests", value=int(result.fun))
        st.metric(label="Normal loan", value=int(result.x[0]))
        st.metric(label="Loan protection", value=int(result.x[1]))
    dfa, dfb = print_func(result.x, help1, help2, help3, euribor2, euribor3, euribor4, extra, cap, margin,m)
    
    def highlight_greaterthan(s, threshold, column):
        is_max = pd.Series(data=False, index=s.index)
        is_max[column] = s.loc[column] >= threshold
        return ['background-color: yellow' if is_max.any() else '' for v in is_max]
    
    if dfa.shape[0]>1:
        for i in ['Interest Paid','Principal Paid','Principal Remain']:
            dfa[i] = pd.to_numeric(dfa[i], errors='coerce')
        dfa = dfa.append(dfa.sum(numeric_only=True), ignore_index=True)
        dfa.iloc[-1,2]=dfa.iloc[-2,2]-dfa.iloc[-2,1]
    if dfb.shape[0]>1:
        for i in ['Interest Paid','Principal Paid','Principal Remain']:
            dfb[i] = pd.to_numeric(dfb[i], errors='coerce')
        dfb = dfb.append(dfb.sum(numeric_only=True), ignore_index=True)
        dfb.iloc[-1,2]=dfb.iloc[-2,2]-dfb.iloc[-2,1]
    col1, col2  = st.columns(2)
    with col1:
        st.header("Normal Loan overview")
        st.dataframe(dfa.style.apply(highlight_greaterthan, threshold=1500, column='Principal Paid', axis=1), 1000,2000)
    with col2:
        st.header("Protected Loan overview")
        st.dataframe(dfb.style.apply(highlight_greaterthan, threshold=1500, column='Principal Paid', axis=1), 1000,2000)