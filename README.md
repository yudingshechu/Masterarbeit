# Masterarbeit: Robust Statistical Learning for Food Security Forecasting
Here are part of the code and current sample of my master thesis. This project/thesis will be finished with R and Python, using Uganda National Household Survey (UNHS) data and other open source data. 

My master thesis focuses on finding a robust Machine Learning model to predict the food insecurity events when facing a shock, such as the breakout of Covid and the war in Ukraine. 

## FCS data
Food Consumption Score (FCS) is widely used in food security study. In this thesis, I generated FCS with the UNHS data by myself, then composition and distribution of FCS could be found in my thesis sample `Master_Thesis.pdf`. 

## Climate data 
Climate data are exported from [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/), and the Uganda map data are downloaded from [HDX--Uganda - Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-uga?). The precipitation and 2 meters range temperature are used. In addition, the rainfall data provided by [VAM Food Security Analysis](https://dataviz.vam.wfp.org/seasonal_explorer/rainfall_vegetation/visualizations) is also used. 

## Nightlight Data 
Nightlight data is a good proxy for economic variables in developling countries. Because the nightlight original data is too large, please download them by yourself in [Earth Observation Group](https://eogdata.mines.edu/products/vnl/), the Monthly Cloud-free DNB Composite `.vcm` version are used. Meanwhile, the nightlight extraction R code is inspired by [this blog](https://berenger.baospace.com/nightlights-satellite-data-free-download/). Notably, Uganda is a tropical country, therefore, I have to combine the two noghtlight file `00N060W.tiff` and `75N060W.tiff` to obtain the whole Uganda data (using `merge()` in R).  The `GSEC1.dta` is part of UNHS data, therefore, it will not be uploaded here. 

In addition, `GSEC1.dta` contains the household ID, their location information, the date of interview and some other basic information. 

## Conflict Data 
Uganda conflict data from 2016 to 2020 are obtained from [Armed Conflict Location & Event Data Project (ACLED)](www.acleddata.com). The number of each type of conflict for a given district and date are generated as variables. Meanwhile, the total fatalities in a given district and a given data is generated as a variable. 

Reference: 
_Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). “Introducing ACLED: An armed conflict location and event dataset: Special data feature”. Journal of Peace Research, 47(5), 651-660. https://doi.org/10.1177/0022343310378914_

## NDVI data
NDVI (Normalized Difference Vegetation Index) is used as a predictor for food security in the ML context as well. This data is also provided by [VAM Food Security Analysis](https://dataviz.vam.wfp.org/seasonal_explorer/rainfall_vegetation/visualizations), for those districts without data, imputation of Uganda national scale data happens. 

# Credits
**Author:** _Gewei Cao, Msc Economics, University of Bonn_

**Supervisior:** _Dr. Clara Brandi, University of Bonn, IDOS; Dr. Lukas Kornher, ZEF_ 
