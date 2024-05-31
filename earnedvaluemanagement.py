# 
# Copyright (C) 2020
#
# This file is a part of PROPCOT / OPKA
# 
# BNEvidenceBase library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301  USA
#



import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as T

import arviz as az

import matplotlib.pyplot as plt

import datetime


# A function to read data from input file, filename is specified in the running script.
# It passes the extracted data from Excel in array format to build_model()
def project_reader(filename):
    dataframeInformation = pd.read_excel(filename,sheet_name=0)
    projectInformation = np.array(dataframeInformation)
    
    dataframeDefinition = pd.read_excel(filename,sheet_name=1)
    projectDefinition = np.array(dataframeDefinition)    

    dataframeProgress = pd.read_excel(filename,sheet_name=2)
    projectProgress = np.array(dataframeProgress)
    
    dataframeMonthly = pd.read_excel(filename,sheet_name=3)
    projectMonthly = np.array(dataframeMonthly)
    
    MONTH_NUMBER = projectMonthly[0].shape[0]
    monthNames = np.array(dataframeMonthly.columns[1:MONTH_NUMBER].tolist())
    
    project = (projectInformation,projectDefinition,projectProgress,projectMonthly,monthNames)
    return project


# A function to build the Bayesian Network with correct relationships and parameters,
# parameters are taken from the input Excel file.
def build_model(project):

    projectInformation = project[0]
    projectDefinition = project[1]
    projectProgress = project[2]
    projectMonthly = project[3]
    
    WP_NAMES = np.array(projectDefinition[:,0])
    WP_NUMBER = projectDefinition[:,0].shape[0]
    MONTH_NUMBER = projectMonthly[0].shape[0]-1
    
    ACTUAL_COST = projectProgress[:,4]

    CONTROL_MONTH = projectInformation[0][1]
    
    with pm.Model() as model:
        
        #Initialising 
        cll_prob = np.zeros([WP_NUMBER,2])
        cul_prob = np.zeros([WP_NUMBER,2])
        CULVALUES = np.zeros(WP_NUMBER)
        CLLVALUES = np.zeros(WP_NUMBER)
        
        PV_mostprobable = np.zeros([4,WP_NUMBER])
        PV_optimistic = np.zeros([4,WP_NUMBER])   
        PV_pessimistic = np.zeros([4,WP_NUMBER])
        
        MONTHS = [0]*WP_NUMBER
        for x in range(WP_NUMBER):
            MONTHS[x] = [0]*MONTH_NUMBER
        
        RISK = list()
        p = list()
        assignment = list()
        centers = list()
        sds = list()
        center_i = list()
        sd_i = list() 
        PV = list()
        p_ = list()
        cull_assign = list()
        cul_dists = list()
        cul_distsTensor = list()
        COMP = list()
        EV = list()
        AC = list()
        PV_Partial = list()      
        
        SPI = list()
        CPI = list()
        
        #Defining risk factor probabilities
        risk_prob = np.zeros([WP_NUMBER,2,2])
    
        for x in range(WP_NUMBER):
            for y in range(2):
                if projectDefinition[x][y+1] != 0:
                    risk_prob[x][y] = np.array([1-projectDefinition[x][y+1],projectDefinition[x][y+1]])
                else:
                    risk_prob[x][y] = np.array([0,0])

        #Filling in Lower and Upper Limit probabilities and values for Completion
        for x in range(WP_NUMBER):
            if projectProgress[x][2] != 0:
                CLLVALUES[x] = projectProgress[x][2]/100
                cll_prob[x] = [0.1,0.9]
            else:
                cll_prob[x] = [0.9,0.1]
                CLLVALUES[x] = 0
                
            if projectProgress[x][3] != 0:
                CULVALUES[x] = projectProgress[x][3]/100
                cul_prob[x] = [0.1,0.9]
            else:
                cul_prob[x] = [0.9,0.1]
                CULVALUES[x] = 1
        
        #Calculating mean and stdev for PV
        for y in range(4):
            for x in range(WP_NUMBER):
                PV_mostprobable[y] = np.array(projectDefinition[:,3+3*y])
                PV_optimistic[y] = np.array(projectDefinition[:,4+3*y])
                PV_pessimistic[y] = np.array(projectDefinition[:,5+3*y])
        
        PV_mean = (4*PV_mostprobable+PV_optimistic+PV_pessimistic)/6
        PV_sd = (PV_pessimistic-PV_optimistic)/6
    
        PV_mean = np.transpose(PV_mean)
        PV_sd = np.transpose(PV_sd)
        PVSD = PV_sd[0][0]
        
        #Calculating PV values
        for x in range(WP_NUMBER):
            for y in range(2):
                if risk_prob[x][y][1] != 0:
                    RISK.append(pm.Categorical("%s_Risk_%d"%(WP_NAMES[x],y+1),p=risk_prob[x][y]))             
        
        for x in range(WP_NUMBER):    
            if projectDefinition[x][1] == 0:
                PV.append(pm.Normal("PV_%s"%WP_NAMES[x],mu=PV_mean[x][0],sd=PV_sd[x][0]))
                p.append(1)
                centers.append(1)
                sds.append(1)
                assignment.append(1)
                center_i.append(1)
                sd_i.append(1)
                
            if projectDefinition[x][1] != 0:
                if projectDefinition[x][2] != 0:
                    p.append(pm.Dirichlet("p%d"%x,np.array([100*risk_prob[x][0][0]*risk_prob[x][1][0],
                                                            100*risk_prob[x][0][1]*risk_prob[x][1][0],
                                                            100*risk_prob[x][0][0]*risk_prob[x][1][1],
                                                            100*risk_prob[x][0][1]*risk_prob[x][1][1]])))
                    centers.append(pm.Normal("centers%d"%x,mu=PV_mean[x],sd=PV_sd[x],shape=4))
                    sds.append(pm.Uniform("sds%d"%x,0,2000,observed=PV_sd[x],shape=4))                
                if projectDefinition[x][2] == 0:
                    p.append(pm.Dirichlet("p%d"%x,np.array([100*risk_prob[x][0][0],
                                                            100*risk_prob[x][0][1]])))
                    centers.append(pm.Normal("centers%d"%x,mu=PV_mean[x,:2],sd=PV_sd[x,:2],shape=2))
                    sds.append(pm.Uniform("sds%d"%x,0,2000,observed=PV_sd[x,:2],shape=2))

                assignment.append(pm.Categorical("assignment%d"%x,p[x]))
                center_i.append(pm.Deterministic("center_i%d"%x,centers[x][assignment[x]]))           
                if projectDefinition[x][2] != 0:
                    sd_i.append(pm.Deterministic("sd_i%d"%x,sds[x][assignment[x]]))
                if projectDefinition[x][2] == 0:
                    sd_i.append(PVSD)                
                PV.append(pm.Normal("PV_%s"%WP_NAMES[x],mu=center_i[x],sd=sd_i[x]))           
            
            p_.append(pm.Dirichlet("p_%d"%x,np.array([10*cul_prob[x][0]*cll_prob[x][0],10*cul_prob[x][1]*cll_prob[x][0],10*cul_prob[x][0]*cll_prob[x][1],10*cul_prob[x][1]*cll_prob[x][1]])))
            cull_assign.append(pm.Categorical("cull_assign%d"%x,p_[x]))
        
        #Splitting PV in months according to data taken from Excel
        #and calculating partial PVs so far -using the month project control is done
        for x in range(WP_NUMBER):
            for y in range(MONTH_NUMBER):
                MONTHS[x][y] = PV[x]*projectMonthly[x][y+1]
        
        for x in range(WP_NUMBER):
            if projectProgress[x][1] != 0:
                PV_Partial.append(pm.Deterministic("Partial_PV_%s"%WP_NAMES[x],np.sum(MONTHS[x][:CONTROL_MONTH])))
            else:
                PV_Partial.append(0)
                
        #First steps to calculate COMP % values according to limits
        for x in range(WP_NUMBER):
            cul_dists.append(list())
            if projectProgress[x][1] == 0:
                cul_dists[x].append(0)
                cul_dists[x].append(0)
                cul_dists[x].append(0)
                cul_dists[x].append(0)
            if projectProgress[x][1] == 100:
                cul_dists[x].append(1)
                cul_dists[x].append(1)
                cul_dists[x].append(1)
                cul_dists[x].append(1)
            if projectProgress[x][1] != 0 and projectProgress[x][1] != 100:
                cul_dists[x].append(pm.Beta("cul_dists%d%d"%(x,0),alpha=projectProgress[x][1]/10,beta=(100-projectProgress[x][1])/10))
                cul_dists[x].append(pm.Bound(pm.Normal,lower=0,upper=CULVALUES[x])("cul_dists%d%d"%(x,1),mu=cul_dists[x][0],sd=1))
                cul_dists[x].append(pm.Bound(pm.Normal,lower=CLLVALUES[x],upper=1)("cul_dists%d%d"%(x,2),mu=cul_dists[x][0],sd=1))
                cul_dists[x].append(pm.Bound(pm.Normal,lower=CLLVALUES[x],upper=CULVALUES[x])("cul_dists%d%d"%(x,3),mu=cul_dists[x][0],sd=1))
            cul_distsTensor.append(T.as_tensor_variable(cul_dists[x]))
        
        #Calculating COMP % --and preparing a dummy probability distribution for AV observations
        for x in range(WP_NUMBER):     
            COMP.append(pm.Deterministic("COMPLETION_%s"%WP_NAMES[x],cul_distsTensor[x][cull_assign[x]]))            
            if projectProgress[x][1] != 0:
                EV.append(pm.Normal("EV_%s"%WP_NAMES[x],mu=PV[x]*COMP[x],sd=sd_i[x]))
            else:
                EV.append(0)
            AC.append(pm.Normal("AC_%s"%WP_NAMES[x],mu=ACTUAL_COST[x],sd=sd_i[x],observed=ACTUAL_COST[x]))
        
        #Performance Indices, SPI and CPI
        for x in range(WP_NUMBER):
            if projectProgress[x][1] != 0:
                SPI.append(pm.Deterministic("SPI_%s"%WP_NAMES[x],EV[x]/PV_Partial[x]))
                CPI.append(pm.Deterministic("CPI_%s"%WP_NAMES[x],EV[x]/AC[x]))
            else:
                SPI.append(0)
                CPI.append(0)
                
                
        PSPI = pm.Deterministic("SPI_PROJECT",np.sum(EV)/np.sum(PV_Partial))
        PCPI = pm.Deterministic("CPI_PROJECT",np.sum(EV)/np.sum(AC))
        
        TEAC = pm.Deterministic("TEAC",(MONTH_NUMBER*30)/PSPI)        
        EAC = pm.Deterministic("EAC",np.sum(PV)/PCPI)
        ETC = pm.Deterministic("ETC",(np.sum(PV)-np.sum(EV))/PCPI)

    return model

# A function to run the sampling algorithm to estimate/calculate the model.
def sample_model(model):
    with model:
        #new
        #trace=pm.sample(njobs=1)
        trace = pm.sample(cores=1)
    return trace

# A function to display results (summary statistics and posterior probability graphs of all nodes) on screen
def display_posterior(trace,filename):
    prj = project_reader(filename)
    WP_NAMES = np.array(prj[1][:,0])
    WP_NUMBER = prj[1][:,0].shape[0]
    PV_names = list()
    PVpartial_names = list()
    EV_names = list()
    COMP_names = list()
    SPI_names = list()
    CPI_names = list()
    Index_names = ["SPI_PROJECT","CPI_PROJECT","ETC","EAC","TEAC"]

    RISK_names = list()
    projectDefinition = prj[1]
    
    for x in range(WP_NUMBER):
        for y in range(2):
            if (projectDefinition[x][y+1]!=0):
                rname = projectDefinition[x][0]+"_Risk_%d"%(y+1)
                RISK_names.append(rname)
                
    for x in range(WP_NUMBER):
        PV_names.append("PV_%s"%WP_NAMES[x])
        PVpartial_names.append("Partial_PV_%s"%WP_NAMES[x])
        EV_names.append("EV_%s"%WP_NAMES[x])
        COMP_names.append("COMPLETION_%s"%WP_NAMES[x])
        SPI_names.append("SPI_%s"%WP_NAMES[x])
        CPI_names.append("CPI_%s"%WP_NAMES[x])
    all_names = RISK_names + PV_names + PVpartial_names + EV_names + COMP_names + SPI_names + CPI_names + Index_names
  
    print(pm.summary(trace,varnames=all_names,stat_funcs=[trace_mean,trace_sd,trace_quantiles]))
    pm.plot_posterior(trace,varnames=all_names)

def excel_posterior2(trace,filename):
    
    #Need to read the data again to set activity number and names
    prj = project_reader(filename)
    WP_NAMES = np.array(prj[1][:,0])
    WP_NUMBER = prj[1][:,0].shape[0]
    
    PV_names = list()
    PVpartial_names = list()
    EV_names = list()
    COMP_names = list()
    SPI_names = list()
    CPI_names = list()
    Index_names = ["SPI_PROJECT","CPI_PROJECT","ETC","EAC","TEAC"]
    
    RISK_names = list()
    projectDefinition = prj[1]
    
    for x in range(WP_NUMBER):
        for y in range(2):
            if (projectDefinition[x][y+1]!=0):
                rname = projectDefinition[x][0]+"_Risk_%d"%(y+1)
                RISK_names.append(rname)
    
    for x in range(WP_NUMBER):
        PV_names.append("PV_%s"%WP_NAMES[x])
        PVpartial_names.append("Partial_PV_%s"%WP_NAMES[x])
        EV_names.append("EV_%s"%WP_NAMES[x])
        COMP_names.append("COMPLETION_%s"%WP_NAMES[x])
        SPI_names.append("SPI_%s"%WP_NAMES[x])
        CPI_names.append("CPI_%s"%WP_NAMES[x])
    all_names = RISK_names + PV_names + PVpartial_names + EV_names + COMP_names + SPI_names + CPI_names + Index_names
    
    outputName = filename+"Output.xlsx"
    traceName = filename+"Trace.xlsx"

    
    summary = az.summary(trace, var_names=all_names,stats=[az.mean, az.sd, custom_quantile_func])
    summary.to_excel(outputName, sheet_name="Summary")

    #pm.summary(trace,var_names=all_names,stat_funcs=[trace_mean,trace_sd,trace_quantiles]).to_excel(outputName,sheet_name="Summary")

    #pm.plot_posterior(trace,var_names=all_names)

    #pm.trace_to_dataframe(trace).to_excel(traceName,sheet_name="Trace")
def excel_posterior(trace, filename):
    # Need to read the data again to set activity number and names
    prj = project_reader(filename)
    WP_NAMES = np.array(prj[1][:, 0])
    WP_NUMBER = prj[1][:, 0].shape[0]
    
    PV_names = []
    PVpartial_names = []
    EV_names = []
    COMP_names = []
    SPI_names = []
    CPI_names = []
    Index_names = ["SPI_PROJECT", "CPI_PROJECT", "ETC", "EAC", "TEAC"]
    
    RISK_names = []
    projectDefinition = prj[1]
    
    for x in range(WP_NUMBER):
        for y in range(2):
            if projectDefinition[x][y + 1] != 0:
                rname = projectDefinition[x][0] + "_Risk_%d" % (y + 1)
                RISK_names.append(rname)
    
    for x in range(WP_NUMBER):
        PV_names.append("PV_%s" % WP_NAMES[x])
        PVpartial_names.append("Partial_PV_%s" % WP_NAMES[x])
        EV_names.append("EV_%s" % WP_NAMES[x])
        COMP_names.append("COMPLETION_%s" % WP_NAMES[x])
        SPI_names.append("SPI_%s" % WP_NAMES[x])
        CPI_names.append("CPI_%s" % WP_NAMES[x])
    
    all_names = RISK_names + PV_names + PVpartial_names + EV_names + COMP_names + SPI_names + CPI_names + Index_names
    
    #outputName = filename + "Output.xlsx"
    #traceName = filename + "Trace.xlsx"

    #new
    #time+name
    now = datetime.datetime.now()
    # Định dạng thời gian
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    outputName = timestamp + "_Output.xlsx"
    traceName = timestamp + "_Trace.xlsx"
    
    # Use ArviZ to generate the summary
    summary_df = az.summary(trace, var_names=all_names)
    
    # Calculate and add custom quantiles to the summary DataFrame
    for var in all_names:
        if var in trace.varnames:
            quantile_df = trace_quantiles(trace[var])
            summary_df.loc[var, '5%'] = quantile_df['5%'].values[0]
            summary_df.loc[var, '50%'] = quantile_df['50%'].values[0]
            summary_df.loc[var, '95%'] = quantile_df['95%'].values[0]
    
    # Save the summary to an Excel file
    summary_df.to_excel(outputName, sheet_name="Summary")
    
    #CODE CŨ
    pm.trace_to_dataframe(trace).to_excel(traceName,sheet_name="Trace")

    # Plot posterior distributions
    # Lấy thời gian hiện tại
    now = datetime.datetime.now()
    # Định dạng thời gian
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    az.plot_posterior(trace, var_names=all_names)
    plt.savefig('img_result/posterior_distribution_'+timestamp+'.png')
    #Không vẽ, chỉ lưu dữ liệu
    #plt.show()



    # Save the trace to a DataFrame and then to an Excel file
"""     trace_df = az.from_pymc3(trace).posterior.stack(draws=("chain", "draw")).reset_index().to_dataframe()
    trace_df.to_excel(traceName, sheet_name="Trace") """


def trace_quantiles2(x):
    #quantiles = np.percentile(x, [5, 50, 95])
    #return pd.DataFrame(quantiles.reshape(-1, 1), index=[5, 50, 95], columns=['quantiles'])
    #return pd.DataFrame(quantiles.reshape(1, -1), index=['quantiles'], columns=[5, 50, 95])
    #return pd.DataFrame(quantiles, index=[5, 50, 95], columns=['quantiles'])

    #new
    #return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))

    # Calculate the 5th, 50th, and 95th percentiles
    quantiles = np.percentile(x, [5, 50, 95])
    quantile_df = pd.DataFrame({
        '5%': [quantiles[0]],
        '50%': [quantiles[1]],
        '95%': [quantiles[2]]
    })
    return quantile_df

def trace_quantiles(x):
    quantiles = np.percentile(x, [5, 50, 95])
    return pd.DataFrame({
        '5%': [quantiles[0]],
        '50%': [quantiles[1]],
        '95%': [quantiles[2]]
    })

    

def trace_sd(x):
    return pd.Series(np.std(x, 0), name='sd')

def trace_mean(x):
    return pd.Series(np.mean(x), name='mean')
