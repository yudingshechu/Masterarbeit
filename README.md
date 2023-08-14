# Masterarbeit: Robust Machine Learning for Food Security Forecasting
Here are the code and the main body of my master's thesis. This project/thesis will be finished with R and Python, using Uganda National Household Survey (UNHS) data and other open-source data. The UNHS data will not be uploaded, and other data are available on the internet. 

## Abstract 
The food insecurity issue is serious in eastern Africa, facing the shocks such as the Covid pandemic and war in Ukraine, a robust machine learning is needed for food insecurity forecasting. This study combines the Uganda National Households Survey data and other open-source data to predict household food insecurity before and during Covid. It is shown that models based on decision trees behave robustly when facing the shock of Covid. These tree-based models provide flexibility for policymakers to trade off the cost and benefit of food insecurity aiding. We also find that demographic and asset features provide the most prediction power. Finally, tree-based methods are robust against limited features or un-updated training data when facing a shock, implying they are robust in practical scenarios.

## Included Algorithms
**Regularized logistic regression**, **support vector machine**, **random forest**, **XGBoost**. In addition, **SHAP** is also used for model interpretation. 

# Response Variable and Predictors
## FCS data (response variable) 
Food Consumption Score (FCS) is widely used in food security studies. In this thesis, I generated FCS with the UNHS data by myself, the composition and distribution of FCS could be found in my thesis sample `Master_Thesis.pdf`. FCS here is the household food security indicator, if FCS is equal to or less than 21, then the household is food insecure. Less than 5% of observations in our data are food insecure, thus, the data is imbalanced. 

## Climate data 
Climate data are exported from [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/), and the Uganda map data are downloaded from [HDX--Uganda - Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-uga?). The precipitation and 2 meters range temperatures are used. In addition, the rainfall data provided by [VAM Food Security Analysis](https://dataviz.vam.wfp.org/seasonal_explorer/rainfall_vegetation/visualizations) is also used. 

## Nightlight Data 
Nightlight data is a good proxy for economic variables in developing countries. Because the nightlight original data is too large, please download them by yourself in [Earth Observation Group](https://eogdata.mines.edu/products/vnl/), the Monthly Cloud-free DNB Composite `.vcm` version is used. Meanwhile, the nightlight extraction R code is inspired by [this blog](https://berenger.baospace.com/nightlights-satellite-data-free-download/). Notably, Uganda is a tropical country, therefore, I have to combine the two nightlight files `00N060W.tiff` and `75N060W.tiff` to obtain the whole Uganda data (using `merge()` in R).  The `GSEC1.dta` is part of UNHS data, therefore, it will not be uploaded here. 

In addition, `GSEC1.dta` contains the household ID, their location information, the date of the interview, and some other basic information. 

## Conflict Data 
Uganda conflict data from 2016 to 2020 are obtained from [Armed Conflict Location & Event Data Project (ACLED)](www.acleddata.com). The number of each type of conflict for a given district and date are generated as variables. Meanwhile, the total fatalities in a given district and a given data is generated as a variable. 

Reference: 
_Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). “Introducing ACLED: An armed conflict location and event dataset: Special data feature”. Journal of Peace Research, 47(5), 651-660. https://doi.org/10.1177/0022343310378914_

## NDVI data
NDVI (Normalized Difference Vegetation Index) is used as a predictor for food security in the ML context as well. This data is also provided by [VAM Food Security Analysis](https://dataviz.vam.wfp.org/seasonal_explorer/rainfall_vegetation/visualizations), for those districts without data, imputation of Uganda national scale data happens. 

## UNHS predictors
In UNHS data, several predictors are generated, including demographic variables, assets variables, and welfare variables. 

# Conclusion and Outlook
Many countries around the world are facing the threat of food insecurity, and the breakouts
of the Covid pandemic and war in Ukraine escalate the already immense food security
challenges the world is facing. This underlines how important it is to adequately forecast
food insecurity. ML can be key to improving forecasting substantially. Therefore, the
robustness of ML in food insecurity prediction must be considered. This study explores
the households and open-sourced data to find robust ML models. Our study could be
extended to other eastern African countries without much cost.

With the ex-post model design, an AUC of 0.81 to 0.84 could be achieved, a higher
AUC implies a broader space for policymakers to trade off the cost and benefit of correctly
detecting real food-insecure households. Without elaborated threshold selection,
62% of food insecure households could be detected, and overall 80% households are correctly
classified. Furthermore, tree-based models outperform, because tree-based models
could reveal the high dimensional interactions among features, and this covers the shock
in training data. Meanwhile, resampling techniques do not help much for tree-based
models, but logistic regression is improved. This implies that the balanced weight scheme
is good enough for tree-based models compared with resampling because the balanced
weight scheme emphasizes more on minority class.

In the ex-ante model design, an overtime mean of 0.83 AUC can be achieved, and at the first and second months of the Covid breakout in our data, without further adjustments,
80% of food insecure households and overall households are correctly classified.
Meanwhile, it is shown that tree-based methods are also robust when using the same
pre-shock training data to predict following during-shock data. Furthermore, when only
collecting ten demographic features, the model performances do not deteriorate significantly,
and random forest performs the most similarly to the full feature models. These
imply that given limited available data, random forest is the most robust one. Moreover,
the tree-based models outperform and show good robustness during Covid than the
logistic models, especially at the beginning of the shock. Again this demonstrates the
complicated essence of food security issues, those high-dimensional interactions have to
be considered in future studies. 

In addition, SHAP analysis shows that demographic and asset features are important
in food insecurity prediction, which confirms that food insecurity is strongly related to
poverty and development. The contributions of micro household data keep consistent
with other studies. In a study of regional food insecurity, those macro open-sourced data
will contribute more. SHAP analysis also gives us policy implications, that one can design
a short questionnaire with the help of SHAP analysis, to collect data and predict food
insecurity economically and effectively.

In conclusion, this study shows that ML, especially tree-based methods are suitable for
food insecurity prediction, because they could reflect the complex interactions of features,
and could better reflect the heterogeneity in data. Tree-based methods are robust to
extreme values, this guarantees them to some extent robustness when facing a shock such
as Covid in future data, because shocks may bring more extreme values. Thus, compared
with the famous logistic regression, tree models such as random forest and XGBoost are
recommended for future research, and for a limited data availability case, random forest
is more robust.

Further studies with different model designs and predictors or an analytical framework for
shock robustness ML analysis could be the next step. For policymakers and international
organizations, on the one hand, ML classification metrics give them more space to trade
off the cost and benefit for food insecure aiding, which shows the great potentiality of
ML implementation on food insecurity. On the other hand, interpretable ML analysis
such as SHAP enables them to select predictors more efficiently and also provides new
insight in confirming the existing food insecurity theories. Finally, our study also shows
an alternative approach on crisis study and gives researchers a data-driven option by ML.

# Credits
**Author:** _Gewei Cao, MSc Economics, University of Bonn_

**Supervisior:** _Dr. Clara Brandi, University of Bonn, IDOS; Dr. Lukas Kornher, University of Bonn, ZEF_ 
