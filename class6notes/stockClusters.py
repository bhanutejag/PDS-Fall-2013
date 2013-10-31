import pandas as pd
import pandas.io.data as web
import numpy as np
import datetime as datetime
from sklearn import cluster

#short list, longer list below
stockList = ["KO","FDX","MMM","DRI","AKAM","COP","ALL","AFL","AMZN", "LLY","HD","TGT","TRV","F",
"HES","JPM","CSCO","CVX","ORCL","OXY","NVDA","LEN","LOW","M","PFE","URBN","XOM","USB","JNJ",
"JDSU","AVB","VNO","AIV","SYMC","SPG","DIS","PEP","MO","AXP","AMGN","PRU","QCOM","SLB","BAC",
"SNDK","SHW","SBUX","GPS","MRK","PHM","UPS","YHOO","MCD","FSLR"]

sp500components = pd.read_csv("c:/dev/sp500_components_20131030.csv")
sp500components.index=sp500components['ticker']
stockDetails = pd.DataFrame(stockList, index=stockList)
stockDetails  = pd.merge(stockDetails, sp500components, how='left', left_index=True, right_index=True)

#full list, comment this out if you are short on running time
stockList = sp500components['ticker'].tolist()

startDate = datetime.datetime(2002,1,1)
endDate = datetime.datetime(2013,10,29)

allData = dict()

#pull data from Yahoo finance and store in dictionary
for thisTicker in stockList:
  thisData = web.DataReader(thisTicker, "yahoo",startDate,endDate)
  thisData['ret'] = np.insert(np.diff(np.log(thisData["Adj Close"].tolist())), 0, [0])
  allData[thisTicker] = thisData

#merged DataFrame, throw out tickers with too many missing values
notSkipped = []
mergedRetData = None
for i in range(len(stockList)):
  thisTicker = stockList[i]
  thisRetData = allData[thisTicker][["ret"]]
  oldColnames = thisRetData.columns.tolist()
  newColnames = oldColnames
  newColnames[thisRetData.shape[1]-1] = thisTicker
  thisRetData.columns = newColnames
  if(i==0):
    mergedRetData = thisRetData
  else:
    mergedRetData = pd.merge(mergedRetData, thisRetData, how='outer', left_index=True, right_index=True)
  nanIdx = np.argwhere(np.isnan(mergedRetData[thisTicker].tolist())).ravel()
  if(len(nanIdx)>0):
    if(len(nanIdx)>30):
      print "SKIPPING %s, has %d NaN return values " % (thisTicker, len(nanIdx))
      mergedRetData = mergedRetData.drop(thisTicker,1)
      continue
    else:
      print "Setting %d NaN return values to 0 for %s" % (len(nanIdx), thisTicker)
      mergedRetData[thisTicker][nanIdx] = 0
  notSkipped.append(i)

stockList = [stockList[i] for i in notSkipped]


covmat = mergedRetData.cov()	#covariance matrix
cormat = mergedRetData.corr()	#correlation matrix

thisKMeans = cluster.KMeans(50)	#50 clusters
thisKMeans.fit(cormat)
clusterDistances = thisKMeans.transform(cormat)	#distances from each center

clusterDistances[0,:]
clusterDistances[1,:]

stockClusters = pd.DataFrame(thisKMeans.predict(cormat), index=stockList)
stockClusters.columns=['cluster']
stockClusters = pd.merge(stockClusters, sp500components, how='left', left_index=True, right_index=True)
stockClusters.sort("cluster")

outData = stockClusters.sort("cluster")
outData.to_csv("c:/dev/stocks_clustered.output.csv")
