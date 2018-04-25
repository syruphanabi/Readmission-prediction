##################################################################Instruction:
# Files "readmission.R" and "ADMISSIONS.csv" should be in the same folder.
# In R console, source("readmission.R").
# Then we can get a new file "readmission.csv".
###############################################################################
#identify readmission events and add labels
rm(list = ls())
da=read.csv("ADMISSIONS.csv")
dat=da
da=da[,c(1:5)]
da=da[order(da[,2],da[,4]),] #Group the table by patients.For each patient, sort the records by discharge time. 
da[,6]=0
len=length(levels(factor(da[,2])))

for(k in c(0:22)){
  
  ptm <- proc.time()
  # for each patient
  for (i in levels(factor(da[,2]))[(2000*k+1):(2000*(k+1))]){
    da1=da[da[,2]==i,] 
    l=length(da1[,1])
    if(l>1){
      for(j in 1:(l-1)){
        # For the i th patient the j th record, 
        # if (admission time of (j+1) th record - discharge time of j th record) <= 30 days, 
        # the readmission label of j th record =1; otherwise the readmission label of j th record =0.
        if ((as.Date(da1[j+1,4])-as.Date(da1[j,5]))<=30)
          da[da[,1]==da1[j,1],6]=1
      }
    }
    
  }
  print((proc.time()-ptm)[3])
}

ptm <- proc.time()
for (i in levels(factor(da[,2]))[(2000*23+1):len]){
  da1=da[da[,2]==i,]
  l=length(da1[,1])
  if(l>1){
    for(j in 1:(l-1)){
      if ((as.Date(da1[j+1,4])-as.Date(da1[j,5]))<=30)
        da[da[,1]==da1[j,1],6]=1
    }
  }
}
print((proc.time()-ptm)[3])

dat=dat[order(dat[,2],dat[,4]),]
dat[,20]=da[,6]
write.csv(dat,"readmission.csv",row.names = FALSE)



