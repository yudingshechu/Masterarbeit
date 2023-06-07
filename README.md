# Masterarbeit: Robust Statistical Learning for Food Security Forecasting
Here are part of the code and current sample of my master thesis. This project/thesis will be finished with R and Python, using Uganda National Household Survey (UNHS) data and other open source data. 

My master thesis focuses on finding a robust Machine Learning model to predict the food insecurity events when facing a shock, such as the breakout of Covid and the war in Ukraine. 

## FCS data
Food Consumption Score (FCS) is widely used in food security study. In this thesis, I generated FCS with the UNHS data by myself, then composition and distribution of FCS could be found in my thesis sample `Master_Thesis.pdf`. 

## Climate data 
Climate data are exported from [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/), and the Uganda map data are downloaded from [HDX--Uganda - Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-uga?). The precipitation and 2 meters range temperature are used. 

## Nightlight Data 
Nightlight data is a good proxy for economic variables in developling countries. Because the nightlight original data is too large, please download them by yourself in [Earth Observation Group](https://eogdata.mines.edu/products/vnl/), the Monthly Cloud-free DNB Composite `.vcm` version are used. Meanwhile, the nightlight extraction R code is inspired by [this blog](https://berenger.baospace.com/nightlights-satellite-data-free-download/). Notably, Uganda is a tropical country, therefore, I have to combine the two noghtlight file `00N060W.tiff` and `75N060W.tiff` to obtain the whole Uganda data (using `merge()` in R).  The `GSEC1.dta` is part of UNHS data, therefore, it will not be uploaded here. 

In addition, `GSEC1.dta` contains the household ID, their location information, the date of interview and some other basic information. 

# Credits
**Author:** _Gewei Cao, Msc Economics, University of Bonn_

**Supervisior:** _Dr. Clara Brandi, University of Bonn, IDOS; Dr. Lukas Kornher, ZEF_ 
