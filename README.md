# New York Taxi Fare Analysis

## Introduction
This project is based on the New York Taxi Fare Analysis Kaggle challenge, revolving around predicting the fare of a taxi ride given a pickup and a drop off location. We are to explore the data and uncover interesting observations about the New York Taxi operations.

## Preprocessing
Looking at the description of the training dataset, we find many anomalities data points. They include: 
- Some fare amounts are negative and some are too high (up to 150-200usd), which we think might be wrongly input data. We will omit these samples. About the minimum fare values, we will have it to be 2.5usd (instead of 0usd), since this is the base amount of the fare.
- Some values of longtitude and latitude are too large or too small, according to google map, not pointing to New York at all. We will omit these samples by defining a bounding box for NYC. However, rather than cropping the bounding box from the map, we would use the max and min values of the coordinates from the testing dataset. This way, we make sure to cover all possibles values in the testing dataset. 
- We also check if there is any null value. Turns out to be very many. We also omit these samples.

## Exploring dataset
### Correlation
We will compute the Pearson correlation between the following variables:
* Euclidean distance of the ride and the taxi fare: 0.8669
* Time of day and distance traveled: -0.0180
* Time of day and the taxi fare: -0.0314

We see that the highest correlation is between the distance and the fare amount, which seems obvious. The second and third correlations are not high at all. Maybe because they don't have a strong linear relationship with the fare amount and ride distances. However, they still may have a relationship in another form. We will later plot the figures between these features against fare amount to have a better view.

### Correlation plottings
Now we plot the scatter figure of each pair of features to visualize the correlation values. Here, for the time of the day feature, we will make it finer (to minutes) by interpolating the middle points. 

<p align="center">
 <img src="../master/illustrations/pic1.JPG">  
</p> 

Apparently , we find quite a linear relationship between the distance and fare amount feature. For the other two correlations, although that we don't see a linear relationship, but we still find a relationship between the two variables. Indeed, the plots show that the time of day does affect the distance as well as the fare amount. To be particular, in the time between 12am - 7am, there is a decrease in both distance and fare amount. This can be explained intuitively by the fact that people are sleeping at the time. Therefore, we should take the feature of time of day into consideration when predicting fare amount. Besides, we also find out that, the fare price in the morning and at night are different, according to this ![site](https://www1.nyc.gov/nyc-resources/service/1271/yellow-taxifares).
On the other hand, we find something very interesting in the three plots. That there are many completely aligned straight strokes of points on the plots. This might be because of one common ride that many people take. Based on three plots, we guess that this ride usually costs about $50, and about 15 miles long. 
We try to find out what are these "hot-spots" by filtering out these data points and investigate to see where the ride go from/to. It turns out that all of these pick-up/drop-off points cluster at the JFK airport, which is quite far from the downtown. Checking on the internet, we find that there is a "fixed" fare amount to go from/to JFK to/from Manhattan, which is about 51$. So this likely to be the reason for those abnormal straight clusters in the scatter plot.

### Map visualizations
Now we will play around, trying stuffs to see if there are any interesting insights from the training dataset. First, to easily visualize all the points, we will plot all of them on a map of NYC. Using the map image here: 'https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png
Plotting the figure, it reveals our guessing before about the ride from/to JFK. It is indeed that there are many
rides which shows a dense points cloud on the map at the JFK location.
<p align="center">
 <img src="../master/illustrations/pic2.png">  
</p> 

### Relationship between year and fare amount

Playing around, I also find one interesting thing about the relationship between the fare amount and the year the rides happened. The figure below illustrates this observation. We find that the prices increase over time. Indeed there is a drastically increase in fare amount before and after the year of 2012. There seems to be a change in policy of taxi companies around this point of time. 
<p align="center">
 <img width="500" src="../master/illustrations/pic3.JPG">  
</p>

### Relationship between ride direction and fare amount

Another interesting point we found out is that the relationship between direction of rides (in terms of angles) and the fare amount. The code below implements a function for calculating the angles. Following is a plot of the direction against fare amount. We see that there are many rides running at about 55 and -55 degrees. My guess is this is the main street direction of Manhattan, and therefore the road is usually much longer, hence the higher taxi fare. We can use this clue to predict the fare price. Furthermore, rides on these directions (~55 and -55 degrees) have the most correct distance computation because they just go straight. This added information might help the model learn better .
<p align="center">
 <img width="500" src="../master/illustrations/pic4.JPG">  
</p> 

## Building model
According to our observations in quesiton 4, we find there are ralationships between the year, time of day the ride occurs, the direction and the fare. Therefore, we will include them into the selected features set. Other features we will use are "distance", and "number of passengers". The "distance" feature seems obvious enough to be included in the features set. For the "number of passengers" feature, although we have not seen an obvious relationship with the fare amount now. However , it might be a latent relationship (maybe
through some transformations). Therefore, we will still include it in the feature set. In conclusion, the set of features we will use for model is: 
<p align="center">
 <i>[year , time of day , distance, number of passengers, direction]
</p> 

Please refer to pdf report file for detailed results.
