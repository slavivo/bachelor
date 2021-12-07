import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys
from os import path, getcwd
import sys

THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from device_api.Argument_Parser import File_Draw_Arg_Parser

#log to test.csv - python3 device_log.py -s /dev/ttyACM0 -a 4 -o test.csv
#draw from csv - python3 file_draw.py -i 'test.csv' -x time_ms -y accelerationX_g

arg_parser = File_Draw_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

####################################### Read Input File #################################################################################
def readInputFile():
    global options
    global verbose
    delim = options.delimiter
    fileName = options.input
    verbose = options.verbose
    global df
    if(delim == None):
        if(verbose==True):
            print("Delimiter is not specified using \";\"")
        delim = ";"
    if(fileName[-4:]==".csv"):
        try:
            df = pd.read_csv(fileName,delim)
        except FileNotFoundError:
            print("No csv file with name \""+fileName+"\" found")
            exit()
    else:
        if(fileName[-3:]==".h5"):
            fl = h5py.File(fileName, "r")
            dataDic={}
            for key in fl.keys():
                dataDic[key]=fl.get(key)
            df =  pd.DataFrame(dataDic)
            fl.close()
        else:
            print("Unknown format, supported formats: csv,h5")
    if(verbose==True):
        print("Input shape:",df.shape)
        print("Available data series names:"+str(list(df.keys())))


####################################### Get Keys for dataframe ################################################################################
def getKeysForDataframe():
    global y
    global y1
    global y2
    global x
    if(options.y_series_number!=None):
        yNum = int(options.y_series_number)
        keys = df.keys()    
        if(yNum>(len(keys)-1)):
            print("Y series number is too big, max is",(len(keys)-1))
            exit()
        y = keys[yNum]
    else:
        y = options.y_series   

    if(options.y1_series_number!=None):
        y1Num = int(options.y1_series_number)
        keys = df.keys()    
        if(y1Num>(len(keys)-1)):
            print("Y1 series number is too big, max is",(len(keys)-1))
            exit()
        y1 = keys[y1Num]
    else:
        y1 = options.y1_series   

    if(options.y2_series_number!=None):
        y2Num = int(options.y2_series_number)
        keys = df.keys()    
        if(y2Num>(len(keys)-1)):
            print("Y2 series number is too big, max is",(len(keys)-1))
            exit()
        y2 = keys[y2Num]
    else:
        y2 = options.y2_series   

    if(options.x_series_number!=None):
        xNum = int(options.x_series_number)
        keys = df.keys()    
        if(xNum>(len(keys)-1)):
            print("Y series number is too big, max is",(len(keys)-1))
            exit()
        x = keys[xNum]
    else:
        x = options.x_series   
    if(y==None):
        print("Y series is not specified!!!")
        exit()

####################################### Get data to draw by keys ###############################################################################
time=None
data=None
data1=None
data2=None

def loadDataSeries():
    global time
    global data
    global data1
    global data2

    try:
        data = df[y]
    except KeyError:
        print("No data series with name \""+y+"\" found in file")
        exit()
    if(y1!=None):
        try:
            data1 = df[y1]
        except KeyError:
            print("No data series with name \""+y1+"\" found in file")
            exit()
    if(y2!=None):
        try:
            data2 = df[y2]
        except KeyError:
            print("No data series with name \""+y2+"\" found in file")
            exit()
    if(x==None):
        time = range(0,len(data))
    else:
        try:
            time = df[x]
        except KeyError:
            print("No data series with name \""+x+"\" found in file")
            exit()


def checkLengths():
    if(len(time) != len(data)):
        print("X and Y series lengths are not the same")
        exit()
    if(y1!=None):
        if(len(time) != len(data1)):
            print("X and Y1 lengths are not the same")
            exit()
    if(y2!=None):
        if(len(time) != len(data2)):
            print("X and Y2 lengths are not the same")
            exit()

####################################### Plotting data ################################################################################3
def plotData():
    subplot = options.subplot
    if(verbose==True):
        print("Plotting:")
        print("\""+str(x)+"\",length("+str(len(time))+")")
        print("\""+str(y) +"\",length("+str(len(data))+")")
        if(y1!=None):
            print("\""+str(y1) +"\",length("+str(len(data1))+")")
        if(y2!=None):
            print("\""+str(y2) +"\",length("+str(len(data2))+")")
    
    if(subplot):
        ys=[y]
        datas=[data]
        if(y1!=None):
            ys.append(y1)
            datas.append(data1)
        if(y2!=None):
            ys.append(y2)
            datas.append(data2)
        fig, axs = plt.subplots(len(ys))
        for i in range(len(ys)):
            axs[i].plot(time,datas[i],label=ys[i])
            axs[i].grid(True)
            axs[i].legend(loc='upper right', shadow=True, fontsize='medium')
    else:
        plt.plot(time,data,label=y)
        if(y1!=None):
            plt.plot(time,data1,label=y1)
        if(y2!=None):
            plt.plot(time,data2,label=y2)
    if(x==None):
        plt.xlabel("time")
    else:
        plt.xlabel(str(x))
    plt.grid(True)
    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.show()


readInputFile()
getKeysForDataframe()
loadDataSeries()
checkLengths()
plotData()