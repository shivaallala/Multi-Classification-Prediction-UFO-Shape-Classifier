# Multi-Classification-Prediction-UFO-Shape-Classifier-

### Problem Statement

The goal of this project is to classify UFO sightings based on their shapes and identify potential trends or patterns using machine learning.

### Data Source and Background

The dataset for this project is sourced from Kaggle and contains over 80,000 reports of UFO sightings over the last century. The dataset includes city, state, time, description, and duration of each sighting, with two versions available: scrubbed and complete. The complete data includes entries with missing or erroneous information.

**Data Source**: https://www.kaggle.com/datasets/NUFORC/ufo-sightings?resource=download

**Content**

There are two versions of this dataset: scrubbed and complete. The complete data includes entries where the location of the sighting was not found or blank (0.8146%) or have an erroneous or blank time (8.0237%). Since the reports date back to the 20th century, some older data might be obscured. Data contains city, state, time, description, and duration of each sighting.

### Importance of the Question:

**Inspiration**

1. What areas of the country are most likely to have UFO sightings?
2. Are there any trends in UFO sightings over time? Do they tend to be clustered or seasonal?
3. Do clusters of UFO sightings correlate with landmarks, such as airports or government research centers?
4. What are the most common UFO descriptions?

**Importance**

    - Understanding the patterns and classifications of UFO sightings has several important implications:

    - Public Interest and Safety: By identifying patterns in UFO sightings, we can address public curiosity and concerns. If certain shapes or trends correlate with specific times or locations, authorities can better manage public safety and information dissemination.

    - Scientific Inquiry: This research can contribute to the broader scientific community's understanding of unexplained aerial phenomena. By providing a systematic analysis of UFO sightings, we add to the body of knowledge that may inspire further research and technological advancements.

    - Policy Making: Insights from this analysis can aid governmental and defense agencies in resource allocation and preparedness. Recognizing patterns in sightings might help in developing protocols for investigation and response.

    - Cultural Impact: UFO sightings are a part of modern folklore and cultural identity. This project can provide a data-driven narrative that complements anecdotal evidence, enriching cultural and historical contexts.


### Tools Requirements and Techniques Used

1. Tools requirements:
    - CPU, general-purpose processors, Recommend Intel core i7/i9 or better. 
    - Python programming language 
    - Jupyter Notebook  or Jupyter Lab for interactive data analysis and visulization (Google Colab can also work)
    - Integrated Development Environments (IDEs): Such as PyCharm or VSCode for code development and debugging.
    - Install all neccessary python packages (Pandas, scikit-learn, seaborn, numpy, etc...) for analytics and suupervised and unsupervised learning models. 

2. Data Preprocessing:
    - Cleaning and preparing the data.
    - Handling missing values and standardizing formats.

3. Exploratory Data Analysis (EDA):
    - Initial examination to uncover basic patterns, anomalies, and insights.

4. Feature Engineering:
    - Creating relevant features to improve model performance.

5. Machine Learning Models:
    - Multi-Class Classification: Using algorithms like Decision Trees, Random Forest, and Support Vector Machines (SVM).
    - Clustering: Using methods like K-Means and DBSCAN to identify patterns and trends.


### Expected Results

1. Classification Model:
    - Develop a robust classification model to predict UFO shapes based on sighting reports.

2. Trend Analysis:
    - Identify temporal and geographical trends in UFO sightings, such as seasonal patterns and regional hotspots.

3. Correlation Insights:
    - Analyze correlations between UFO sightings and various landmarks or significant locations.


## Data Exploration

Firstly, import all neccessary resources and tools, primarly python packages to preform data transformation and analysis. 

![python packages](./Images/Python%20packages.png)


**General Info** 

![general info](./Images/df.info.png)

**Target feature distribution (UFO shape)**

![Target feature distribution (UFO shape)](./Images/Target%20feature%20distribution%20(UFO%20shape).png)

The chart graphically represents the distribution of various UFO shapes reported in the dataset. The bar chart, displaying the count of sightings for each shape, reveals that the most frequently reported shape is "light," with 16,565 sightings, followed by "triangle" with 7,865 sightings and "circle" with 7,608 sightings. Other commonly reported shapes include "fireball" (6,208 sightings), "other" (5,649 sightings), and "unknown" (5,584 sightings). 

Shapes such as "sphere," "disk," and "oval" also have significant counts, ranging from approximately 3,700 to 5,400 sightings. Less frequently reported shapes, such as "rectangle," "cylinder," and "diamond," have counts between 1,000 and 1,300. Rarely reported shapes, such as "teardrop," "cone," and "cross," have fewer than 800 sightings each. Interestingly, there are several shapes with extremely low counts, including "delta," "round," "crescent," "pyramid," "flare," "hexagon," "dome," and "changed," each having between 1 and 7 sightings.

The accompanying red line plot shows the cumulative number of sightings, which provides a visual indication of how the total number of sightings accumulates as different shapes are added. This combination of bar chart and line plot offers a comprehensive view of both the individual distribution and cumulative effect of UFO shape sightings in the dataset.


## Initial Data Cleaning and Imputation

To ensure the integrity of the original dataset, a new dataset was created specifically for the cleaning process. The "duration (hours/min)" column was removed due to redundancy, as the duration in seconds was already present. Features were transformed to their respective datatypes, such as converting numeric features to integers or floats and datetime features to datetime objects. The state and country features were analyzed and imputed to address missing values. Leading and trailing spaces were stripped from the data to ensure standardization and consistency. Extreme outliers, particularly in the target variable, were removed. Imputation was performed to address redundant definitions across multiple classes within the target variable. Finally, after these cleaning steps, any remaining rows with NA values were dropped to produce a clean and consistent dataset for further analysis.

To begin the data cleaning process, a new dataset, ufo_clean, was created by copying the original dataset. The redundant "duration (hours/min)" column was dropped. Dates were transformed to the datetime format for the "datetime" and "date posted" columns, ensuring consistency in date handling. The "duration (seconds)" column was converted to a numeric datatype to facilitate analysis, and the "latitude" column was also transformed to a numeric format. These steps ensured that the data types were appropriate for further analysis and cleaned of any inconsistencies. Below illustration decribes the new dataset with the changes.

![ufo_clean](./Images/ufo_clean.png)

**Remove outliers Imputing target feature Shape**

![initial ufo value counts](./Images/ufo_initial%20value%20counts.png)

It is clear from the above information, that there are about 8 shapes that have extremly minimal data points. Due to lack of observations on these shapes, they may be removed from the dataset.This data is not useful for predictive classification. Now we will have removed 8 shapes from the data, and are left with 21 shapes for multi-class classification.

Looking at the left over shapes, we can observe somethign really important. Some shapes have synonymous definitions. It is reasonable to define the shape of an egg as an oval. Yet, in the shape classes, there is a class that represents oval 'shaped' observation and another class is represented as an egg shaped observation. The same goes for circle and disk, cylinder and cigar. 

When it comes to 'unknown' or 'other', based on the comments provided, the shape of the sighting may not have been recorded or was concluded not definitive. There are instances of similar or arbitray obervations. This means, unknown and other can be used interchangeably.

![unknown and other comments](./Images/unkown%20and%20other%20comments.png)

The obervations for changing and formation are also similar when viewing comments on these classes. Both formation and changing shape values describe some sort of inconsistency in the subject of the observation. Both decribes a metamorphose property pertaining to the subject of the observation.

![formation and changing comments](./Images/formation%20and%20changing%20comments.png)

From these observation, let us impute some of the classes and combine with their counterpart. We will selected one class and name and map to the other while maintain all other attributes integrity. Combining synonymous classes in your dataset based on observed similarities is a reasonable idea, particularly when dealing with categorical data where overlapping or synonymous categories can add noise and reduce the clarity of analysis. Below is the reduced and imputed ufo shape feature details.

shape_mapping = {
    'circle': 'disk',
    'egg': 'oval',
    'cylinder': 'cigar',
    'unknown': 'other',
    'formation': 'changing'
}

![cleaned ufo shape feature](./Images/cleaned%20ufo%20shape%20feature.png)


##### Benefits

  - Data Simplification: Reducing the number of classes makes the dataset simpler and more manageable, which can improve the performance of your clustering algorithms and make the results easier to interpret.
  - Increased Sample Size: By merging similar categories, you increase the sample size of the combined class, which can lead to more robust statistical analyses.
  - Reduced Redundancy: Eliminating synonymous categories helps in reducing redundancy in the dataset, which can improve the overall quality of the data.

##### Considerations

  - Loss of Specificity: While combining classes can simplify the data, it might also lead to a loss of specific information that could be valuable. For instance, the distinction between 'disk' and 'circle' might carry some nuances that are lost when combined.
  - Validation: Ensure that the mappings are validated by domain experts if possible. What might seem synonymous to a layperson could have subtle differences in a specialized field like UFO sightings.
  - Consistency: Make sure the mappings are applied consistently across the dataset to avoid introducing new inconsistencies.


**Impute state and county features**

1. To address inconsistencies and missing values in the state and country features of the UFO sightings dataset, several steps were taken. Initially, country codes were mapped to their respective full names (e.g., 'us' to 'United States') to prevent ambiguity with state abbreviations. This mapping ensured clarity in distinguishing between country and state values.

country_mapping = {
    'us': 'United States',
    'ca': 'Canada',
    'gb': 'United Kingdom',
    'de': 'Germany',
    'au': 'Australia'
}

![country distribution](./Images/county%20distribution.png)


2. Upon examining instances where the country was null but the state was populated, it was observed that certain states like 'tx' (Texas) were mistakenly treated as countries. Logical imputation was applied, where if a state belonged to the US or Canada based on predefined lists of state abbreviations and Canadian provinces, it was assigned the corresponding country name.

  - Below is country and state feature where country is NA

![Country is NA](./Images/country%20is%20NA.png)


3. Further investigation revealed inconsistencies in non-US and non-Canada countries where state values were either missing or incorrectly mapped (e.g., 'nc' mapped to 'gb' for United Kingdom). To rectify this, a custom function was developed to correctly map state values to their respective countries. States within these countries were standardized to avoid misinterpretation of the data.

  - Understanding states within non US or Canada countries

![US states mapped to Non-US countries](./Images/US%20states%20mapped%20to%20non%20US%20countries.png)


4. After applying these transformations and imputations, remaining rows with null values were dropped to ensure dataset completeness and integrity. The resulting ufo_clean dataset now contains standardized state and country information suitable for further analysis and modeling, reducing ambiguity and ensuring accurate representation of geographical attributes in UFO sighting reports.

![states mapped](./Images/states%20mapped%20to%20countries.png)


**Duration (seconds) - removing outliers**

![duration distribution](./Images/duration%20distribtuion.png)

The max duration of a sighting is 97836000.0 seconds, the minimum duration is 0.001 seconds and the average duration is 7877.286460723777 seconds. we can see that there are some values for duration that are extreme values which can impact the models and training. We can preform IQR method to rid outliers but the extreme outliers can also impact lower bound and upper bound when calculating 25% and 75% quantiles. Lets rid of extreme outliers using percentile method. Below is the distribution after removing extreme outliers. After removing outliers max duration of a sighting is 7200.0 seconds, the minimum duration is 1.5 seconds and the average duration is 611.0101017590368 seconds. 

![after duration distribution](./Images/after%20duration%20distribution.png)

![After duration dis boxplot](./Images/After%20duration%20dis%20boxplot.png)

**Dataset after initial cleaning and removing remaining null values**

![cleaned dataset info](./Images/cleaned%20dataset%20info.png)

- cleaned dataset statistical summary for numerical data

![stat summary](./Images/stat%20summary%20cleaned%20dataset.png)


## Feature Analysis

After cleaning the UFO sightings dataset and ensuring the integrity of its features, the next step involves exploring potential relationships and trends among these features.


**UFO sightings change over time**

![sighting increase over time](./Images/sighting%20increase%20by%20country%20over%20time.png)

The plot illustrates the temporal trends in UFO sightings across different countries, focusing on the United States compared to other countries where sightings are relatively consistent over time. The trend shows a notable increase in UFO sightings over time in the United States. This could indicate either a genuine increase in sightings or potentially more comprehensive reporting and data collection methods over the years. In contrast, sightings in other countries appear relatively consistent across the years. This observation may stem from several factors, including varying levels of public interest, reporting protocols, or cultural differences in UFO reporting and interpretation. The consistency observed in other countries could also be influenced by the availability of data. If data collection methods or reporting standards differ significantly between countries, it could lead to disparities in the number of recorded sightings.



**Exploring UFO obervations by Cities**

1. Unique Cities: There are 16,833 unique cities recorded in the dataset.

2. Observation Counts:
  - Less than 10 Observations: 15,347 cities fall into this category, indicating a majority of cities have relatively few recorded UFO sightings.
  - More than 100 Observations: Only 56 cities have 100 or more recorded UFO sightings, representing a smaller subset of cities with more frequent sightings.

3. Summary Table: The table summarizes the distribution of cities based on the number of UFO observations:

![sightings by city](./Images/sightings%20by%20city%20table.png)

![city sightings pie chart](./Images/city%20sightings%20pie%20chart.png)

The pie chart visually represents these categories, illustrating the proportion of cities in each observation range. It shows that the vast majority of cities (over 60%) have only 1 recorded UFO observation, highlighting the variability and sparse nature of UFO sighting reports across different cities.

This analysis provides insights into the distribution of UFO sightings across cities, emphasizing the prevalence of cities with few sightings and a smaller number of cities with more frequent UFO observations.


The cities below had the highest reported sightings in the entire dataset:

![cities with highest sightings](./Images/city%20with%20hightest%20sightings.png)

**Below Scatter plots shows the data distribution with respect to relationships between different features**

1. Duration of sightings of different UFO shapes by country

![Duration of sightings of different UFO shapes by country](./Images/Duration%20of%20sightings%20of%20different%20UFO%20shapes%20by%20country.png)

2. latitude and longitude of sightings of different UFO shapes by country

![latitude and longitude of sightings of different UFO shapes by country](./Images/latitude%20and%20longitude%20of%20sightings%20of%20different%20UFO%20shapes%20by%20country.png)

3. Sightings distribution by country

![Sightings distribution by country](./Images/sightings%20by%20country%20.png)

- Due to a high skewed distribution of country datapoints, it will be excluded from modeling.


## Modeling

Before we construct any classification models let us preprocess our data. Preprocessing the UFO dataset is essential to ensure that our data is in a suitable format for machine learning models. Initially, we extracted additional features from the datetime column, specifically the year and month, which can provide valuable temporal insights. The target variable, 'shape', was encoded using LabelEncoder to transform categorical shape labels into numerical values required for model training. For the features used in modeling, we selected 'state' and 'duration (seconds)', ensuring to standardize the numeric feature 'duration (seconds)' using StandardScaler to scale data for consistent model performance. Categorical feature 'state' was encoded using OneHotEncoder to convert state names into binary vectors, preserving categorical information without imposing numerical order.

![traintestsplit](./Images/traintestsplit.png)

Splitting the preprocessed data into training and testing sets is crucial to evaluate model performance on unseen data accurately. By dividing the dataset into 70% training and 30% testing subsets, we ensure that our models can learn patterns from the training data while being evaluated on independent test data to assess generalization. Preprocessing facilitates data standardization and transformation, ensuring that models can effectively interpret and learn from the data without biases from differing scales or formats. Ultimately, preprocessing prepares our UFO dataset for robust machine learning model construction and evaluation, enabling us to make accurate predictions about UFO shapes based on geographical and temporal characteristics.


### RandomForest Classifier

Random Forests stand out as a robust choice for multi-class classification tasks due to their versatile capabilities. They excel in handling complex data structures and managing overfitting through ensemble learning, where multiple decision trees collectively contribute to robust predictions. This approach not only enhances generalization but also provides valuable insights into feature importance, crucial for understanding the driving factors behind each class prediction. Moreover, Random Forests are known for their reliability in diverse datasets, making them a powerful tool for accurate and scalable multi-class classification tasks.

The provided pipeline encapsulates a streamlined approach to machine learning model building in Python, particularly in a Jupyter notebook environment. It integrates both preprocessing and modeling steps into a single coherent workflow. The ColumnTransformer within the preprocessing step allows for simultaneous handling of numerical (StandardScaler) and categorical (OneHotEncoder) features. This ensures that all necessary transformations are applied consistently across different types of data before feeding them into the classifier.

![randomforest pipeine](./Images/randomforest%20pipeline.png)

**Randomforest model classification report**

- Training data accuracy 

![Train score](./Images/randomforest%20train%20scores.png)

  - Precision: The precision measures how many of the predicted instances for each class are actually correct. Here, precision scores are generally low across most classes, indicating a high rate of false positives. For instance, classes like 0, 1, 2, 3, etc., have precision values ranging from 0.07 to 0.67. This suggests that the model's predictions for these classes have a high chance of being incorrect.
  - Recall: Recall measures how well the model captures instances of each class. It is also generally low, with values ranging from 0.01 to 0.71. This indicates that the model misses a significant number of instances of each class.
  - F1-score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure between them. The F1-scores are generally low across classes, ranging from 0.00 to 0.38.
  - Support: Indicates the number of instances of each class in the training set.

- Test data accuracy 

![Test scores](./Images/random%20forest%20test%20scores.png)

  - The pattern in the test set classification report is similar to the training set.
  - Precision, Recall, and F1-score: The metrics show similarly low values across most classes, indicating that the model's performance did not improve significantly when applied to unseen data. Precision values range from 0.00 to 0.22, recall from 0.00 to 0.61, and F1-score from 0.00 to 0.32.
  - Support: Indicates the number of instances of each class in the test set.

Precision is crucial because it tells us how reliable the positive predictions are for each class. In a multi-class classification problem like this, low precision scores indicate that when the model predicts a particular shape, it is often incorrect. This can lead to misclassification and unreliable predictions in real-world applications. Low precision can be problematic, especially if the consequences of misclassification are significant (e.g., in medical diagnoses or financial predictions). Improving precision involves reducing false positives, which often requires refining the model's decision boundaries or addressing class imbalance issues in the dataset.

while the Random Forest model shows some ability to predict classes, the low precision scores indicate a need for further model refinement or consideration of different algorithms or data preprocessing techniques to improve accuracy and reliability. Using GridSearch to explore other alorithms to train our data can help identify the best configuration that can handle this multi-class classification problem. 

## Exploring other algorithms with GridSearch 

The GridSearch process explored six different classification models. 

1. Logistic Regression, KNN, and Decision Tree:

  - These initial algorithms were chosen for their versatility and interpretability. They were run with varying hyperparameters to identify the best configuration that could handle the multi-class classification problem posed by UFO shapes.

![Gs models 1](./Images/GS%20model%201.png)

2. Random Forest Classifier, XGBoost, and SVC (Support Vector Classifier):

  - These algorithms were selected based on their ability to handle complex data interactions, non-linear relationships, and high-dimensional data. Each algorithm was tuned using GridSearch to optimize its performance.

![GS models 2](./Images/GS%20models%202.png)

**Importance of Trying Different Algorithms**

  - Diverse Capabilities: Each algorithm has unique strengths. For instance, Random Forests are robust against overfitting and handle categorical features well. XGBoost excels in boosting weak learners and can capture complex interactions. SVMs are effective in high-dimensional spaces with complex decision boundaries.

  - Performance Variability: The goal was to test if more complex algorithms could better capture the underlying patterns in the UFO dataset that simpler models might miss. By comparing results across different algorithms, we gain insights into which approach might be most effective for this specific classification task.

**Are They Good for Multi-Class Modeling?**

  - Yes, but with nuances: All selected algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, SVC) are capable of handling multi-class classification tasks. They differ in their approach to handling class imbalances, non-linear relationships, and feature interactions.

  - Algorithm Suitability: The suitability depends on factors such as dataset size, feature complexity, and computational resources. For instance, SVMs are memory-intensive but effective in high-dimensional data, while Decision Trees are less computationally intensive but can overfit without proper pruning.

### Model preformance results

![model results df](./Images/model%20results%20df.png)

### Analysis of Model Performance

- RandomForestClassifier:
  - The RandomForestClassifier exhibited moderate performance across various metrics. With parameters set to max_depth=10 and n_estimators=500, it achieved a recall of 0.212 and an accuracy of 0.210. The precision_weighted_score was 0.155, indicating a reasonable balance between capturing patterns in the data and avoiding overfitting. Training time was relatively long at 121.51 seconds, reflecting the ensemble nature of Random Forests and the number of trees involved. This model likely benefited from its ability to handle complex relationships and feature interactions inherent in the dataset, making it a robust choice for multi-class classification tasks where linear separability is limited.

- XGBClassifier:
  - The XGBClassifier, known for its gradient boosting approach, showed performance metrics similar to the RandomForestClassifier but with slightly lower precision. It was optimized with a learning rate of 0.1 and 10 estimators. The model achieved a recall of 0.212 and an accuracy of 0.208, with a precision_weighted_score of 0.121. Training was quicker compared to RandomForestClassifier at 19.51 seconds, suggesting a trade-off between computational efficiency and model complexity. The choice of parameters might not have fully captured the intricate relationships within the data, potentially limiting its predictive power compared to other models.

- SVC (Support Vector Classifier):
  - The SVC struggled with the dataset's high dimensionality and class imbalance despite its capability to define hyperplanes for classification. With a regularization parameter C=0.1, it achieved a recall of 0.213 and an accuracy of 0.209. However, the precision_weighted_score was notably low at 0.067, indicating challenges in accurately predicting all classes. Training time was significantly longer at 614.84 seconds, highlighting the computational intensity of SVMs in high-dimensional spaces. The model's performance suggests limitations in effectively separating classes due to the complexity and distribution of feature data.

- Logistic Regression:
  - Logistic Regression, chosen for its simplicity and interpretability, showed consistent but relatively lower performance across all metrics. Regularized with C=0.01, it achieved a recall of 0.213 and an accuracy of 0.210. However, the precision_weighted_score was the lowest among all models at 0.044, indicating significant challenges in predicting class labels accurately. Training time was relatively short at 11.66 seconds, reflecting the model's computational efficiency but also its limitations in capturing complex non-linear relationships within the data.

- KNN (K-Nearest Neighbors):
  - KNN, a non-parametric method sensitive to local patterns, struggled with scalability and generalization in this dataset. Optimized with parameters leaf_size=10, n_neighbors=1000, and using distance-based weights, it achieved a recall of 0.188 and an accuracy of 0.191. The precision_weighted_score was 0.137, indicating moderate precision in classification. Training time was the longest among all models at 273.39 seconds, reflecting its computational demand in handling distance calculations across a large dataset. Despite its flexibility in capturing local patterns, KNN's performance suffered due to noise sensitivity and suboptimal parameter tuning.

- Decision Tree:
  - The Decision Tree model, characterized by its simplicity and interpretability, showed comparable performance to more complex models. With a shallow tree depth (max_depth=5) and using the Gini impurity criterion, it achieved a recall of 0.212 and an accuracy of 0.210. The precision_weighted_score was 0.173, indicating reasonable precision in classification tasks. Training time was exceptionally short at 1.20 seconds, making it the most computationally efficient model tested. The decision tree's performance suggests effective handling of basic relationships within the data but may have underfitted due to its limited depth, potentially missing more complex patterns.

**Reasons for preformance**

The performance differences among these models stem from several factors:

- Data Imbalance: The dataset likely had imbalanced classes among UFO shapes, challenging models to accurately predict less frequent shapes.

- Feature Complexity: Features such as state, duration, and year may not have adequately captured the underlying patterns distinguishing UFO shapes, impacting model performance.

- Model Suitability: Each algorithm's effectiveness depends on its ability to handle feature interactions, non-linearities, and class imbalances. Some models may excel where others struggle due to inherent assumptions and parameter choices.

**Reasons for Low Scores**

- Data and Feature Considerations:
  1. Data Imbalance: The UFO dataset likely has imbalanced class distributions among UFO shapes. This imbalance can skew model training, leading to poor performance, especially on minority classes.

  2. Feature Relevance: The features used (e.g., state, duration, year) may not sufficiently capture the distinguishing characteristics of UFO shapes. If there are no strong correlations between these features and the target (UFO shapes), models struggle to generalize and predict accurately.

  3. Complex Relationships: UFO shape classification might inherently involve complex, non-linear relationships between features and shapes. Linear models like Logistic Regression may struggle to capture these complexities, while more advanced models like Random Forests and XGBoost can potentially learn these relationships better.

![model preformance](./Images/models%20preformance.png)

![training time](./Images/training%20time%20comparison.png)

**Theoretical Insight**

- Modeling Challenges: UFO shape classification presents challenges typical of real-world classification problems: imbalanced data, complex relationships, and feature relevance. Addressing these challenges involves:

  1. Feature Engineering: Creating more relevant features that better differentiate UFO shapes could improve model performance.

  2. Algorithm Selection: Choosing algorithms that can handle non-linear relationships and class imbalances effectively.

  3. Model Evaluation: Precision, recall, and F1-score provide insights into model performance beyond simple accuracy, especially important when dealing with imbalanced classes.


Choosing the appropriate model involves understanding these nuances and optimizing parameters to achieve the best performance metrics. Further refinement in feature engineering, model tuning, and potentially exploring ensemble methods could enhance classification accuracy and precision for predicting UFO shapes effectively in real-world scenarios. Each model's strengths and weaknesses provide valuable insights into their applicability and performance in handling complex classification tasks. Additionally, exploring insights from the comment feature through NLP techniques could yield new features and improve the predictive capabilities of the model for UFO shapes. Integrating NLP to analyze comment text data opens avenues for deeper understanding and more nuanced classification strategies.

## NLP Modeling

Given the challenges in building a multi-class classification model to predict UFO shapes using existing algorithms, it is evident that the task requires more advanced techniques. The initial models demonstrated limited success, highlighting the complexity and vagueness of the data. This complexity necessitates a shift in approach. To address this, we are now focusing on the comments feature, which contains rich text data. By leveraging Natural Language Processing (NLP) techniques, we aim to extract meaningful insights from these comments. The goal is to transform the unstructured text into structured data, thereby developing new features that can be used in building an improved classification model.

The preprocessing steps involve tokenization, lowercasing, removing punctuation and stopwords, and normalization through stemming or lemmatization. These steps clean and standardize the text, making it suitable for analysis. Once the text data is processed, it will be utilized to uncover patterns and develop features that can enhance the predictive power of the model. By applying NLP techniques, we hope to capture the nuances within the comments that relate to UFO shapes, ultimately creating a robust model capable of more accurate classification. This approach recognizes the value of text data and seeks to harness its potential to overcome the limitations faced by traditional classification algorithms.

Let us start by seperating original data and extract the text data that we require. The process involves creating a new sub-dataset, ufo_nlp, from the cleaned UFO dataset ufo_clean, focusing on the shape and comments columns. The goal is to extract insights from the comments using Natural Language Processing (NLP) techniques to develop new features for building a classification model to predict UFO shapes.

**Text Preprocessing Steps:**

- Tokenization: Splitting the text into individual words or tokens.
- Lowercasing: Converting all text to lowercase for consistency.
- Removing Punctuation: Eliminating non-alphanumeric characters.
- Removing Stopwords: Filtering out common words like "the", "and", "is" that add little meaning.
- Normalization: Using stemming or lemmatization to reduce words to their base form (e.g., "running" to "run").

**Implementation:**

The preprocessing function preprocess_text applies these steps to each comment in the dataset. Necessary resources from the NLTK library are downloaded, and the function is applied to the comments column, resulting in a new column processed_comments that contains the cleaned and processed text.

**Output:**

The resulting dataset ufo_nlp includes three columns: shape, comments, and processed_comments. This processed text will be used to extract features that can be leveraged in building a more accurate classification model.

![processed comments](./Images/processed%20comments.png)

**Intent and Importance:**

The intent behind this preprocessing is to clean and standardize the text data, making it suitable for further analysis and feature extraction. By processing the comments, we aim to uncover patterns and insights that can enhance the predictive power of the model. This step is crucial because textual data often contains valuable information that can significantly improve the performance of machine learning models when properly processed and analyzed. Applying NLP techniques helps in transforming unstructured text into structured data that can be integrated into the predictive modeling process, potentially leading to more accurate predictions of UFO shapes.


**Indentifying most common words in the dataset**

We identified the most common words in the dataset by combining all processed comments into a single string, tokenizing the text, and calculating word frequencies using the Counter class from Python's collections module. The resulting DataFrame revealed about 34,205 unique words or characters. Notably, many words appeared only once, while some, such as the number '44', appeared thousands of times, indicating anomalies that should be addressed.

![frequent words in comments](./Images/nlp%20frequent%20words%20in%20comments.png)

The top words by frequency include '44', 'light', 'object', 'sky', and 'bright'. These frequent terms suggest common themes in the UFO sightings. To refine our text data for modeling, we need to remove anomalies and infrequent words, focusing on relevant and meaningful terms that contribute to the classification task. This step is crucial for improving the accuracy and effectiveness of our NLP-based classification model.


## NLP LogisticRegression with (TF-IDF Vectorizer)

**Logistic Regression**

  - Logistic Regression is selected due to its efficiency in multi-class classification tasks, ease of implementation, and ability to provide interpretable results. It models the probability of the target class based on the input features, allowing us to understand the contribution of each word to the classification decision. Moreover, it performs well with high-dimensional data, which is typical in text classification problems.

**TF-IDF Vectorizer**

  - The TF-IDF vectorizer transforms the text data into a matrix of TF-IDF features, which helps in identifying and quantifying the importance of words in the dataset. By focusing on the term frequency (TF) and inverse document frequency (IDF), it ensures that common but less informative words receive lower weights, while rare but significant words receive higher weights. This transformation is crucial for text classification as it highlights the most relevant features, improving the model's ability to distinguish between different classes based on meaningful text patterns.


We are now creating a sub-dataset (ufo_nlp) to explore the relationship between tokenized text data and UFO shapes. First, we preprocess the comments to transform them into a structured format suitable for analysis. Then, we encode the target variable (UFO shapes) using label encoding.

Next, we utilize the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert the text data into numerical features. This transformation allows us to measure the importance of words in the comments relative to the entire dataset, making it easier to capture the most relevant features for classification. TF-IDF helps us identify significant words by assigning higher weights to words that are frequent in a document but infrequent across the corpus, thus highlighting words that are more likely to carry meaningful information about the UFO shapes.

![Tfidf Vectorizer code](./Images/TFidf_vectorizer%20code.png)

After transforming the text data, we train a Logistic Regression model on the TF-IDF features. Logistic Regression is chosen for its simplicity, interpretability, and effectiveness in handling multi-class classification problems. It helps us understand the linear relationship between the input features (words) and the target variable (UFO shapes).

The model's coefficients are then extracted to create a relationship_df DataFrame, which stores the relationships between the words and each UFO shape class. This DataFrame provides insights into which words are most indicative of each UFO shape. For instance, a positive coefficient for a word under a specific shape category indicates that the presence of that word increases the likelihood of that shape being predicted.

![tfidf relationship df](./Images/tfidf%20relationship%20df.png)

The relationship_df DataFrame provides a detailed view of how each word contributes to predicting each UFO shape. For example, positive and negative coefficients indicate the strength and direction of the relationship between words and shape classes. By analyzing this DataFrame, we can gain insights into the vocabulary that is most associated with each UFO shape, helping us understand the linguistic patterns and features that are most predictive of each category. Overall, this process of leveraging NLP techniques and Logistic Regression allows us to derive meaningful insights from the text data and improve our UFO shape classification model.


**Summary of top strings associated for each UFO Shape**

The top words for each UFO shape reveal significant linguistic patterns that correspond to the physical descriptions of the shapes. For example, words like "changing," "morphing," and "shifting" for the "changing" shape, and "cigar," "cylinder," and "tube" for the "cigar" shape accurately reflect the dynamic and elongated forms respectively. This pattern is consistent across all shapes, with terms like "triangle" and "triangular" for "triangle," "oval" and "egg" for "oval," and "disk" and "saucer" for "disk" indicating the corresponding forms. These word associations suggest that the top words strongly relate to the actual meanings of each UFO shape, influencing the classification by highlighting characteristic descriptors, which could be a potential loophole if these words are overly relied upon in the model without considering broader context.

![Top words 1](./Images/Top%20words%201.png)    ![Top words 2](./Images/Top%20words%202.png)    ![Top words 3](./Images/Top%20words%203.png)    ![Top words 4](./Images/Tops%20words%204.png)    ![Top words 5](./Images/Top%20words%205.png)


### NLP Model Results: 

The results from the classification report highlight the performance of the logistic regression model on predicting UFO shapes from comment data. The overall accuracy of the model is 51%, which means that about half of the predictions made by the model are correct. This overall accuracy metric provides a general sense of the model’s performance but doesn’t tell the whole story.

![model results](./Images/NLP%20model%20results%20df.png)

**Precision, Recall, and F1-Score:**

  - Precision: This measures the accuracy of the positive predictions. For instance, a precision of 0.88 for the "chevron" shape means that when the model predicts a UFO is chevron-shaped, it is correct 88% of the time. High precision indicates that there are few false positives. For our UFO shapes, high precision in certain shapes like "chevron" and "cigar" suggests that when the model makes a positive prediction for these shapes, it is usually correct.

  - Recall: This measures the ability of the model to find all relevant instances. For example, a recall of 0.71 for the "light" shape means that the model correctly identifies 71% of all true instances of the light shape. High recall is crucial when the goal is to capture as many true instances as possible. For UFO shapes, shapes with higher recall like "light" show that the model is better at finding most of the actual occurrences of that shape in the data.

  - F1-Score: This is the harmonic mean of precision and recall, providing a single measure of a model’s performance by balancing the two metrics. The f1-score for each shape, such as 0.70 for "triangle," indicates a good balance between precision and recall. The f1-score is particularly useful when the class distribution is imbalanced, as it provides a more comprehensive measure of accuracy.

**Effect of These Metrics:**

  - High Precision, Low Recall: For shapes like "chevron" and "cigar," high precision but low recall suggests that while the model is accurate when it predicts these shapes, it misses many true instances. This could be due to insufficient examples in the training data or the similarity of these shapes to others.

  - Low Precision, High Recall: For shapes like "light," high recall but lower precision means the model captures most of the true instances but also includes more false positives. This could happen if the features for these shapes are common across many other shapes.

**Robustness and Areas for Improvement:**

  - Model Robustness: Despite the varying metrics, the logistic regression model provides a baseline performance. Its interpretability and efficiency make it robust for understanding which features (words) are significant for each shape.

- Improvement Strategies:

  - Feature Engineering: Enhancing the quality of text preprocessing and including more sophisticated text features can improve model performance.
  - Advanced Models: Trying more complex models like Random Forests, Gradient Boosting Machines, or neural networks could capture non-linear relationships better.
  - Balancing the Data: Addressing class imbalances through techniques like oversampling or undersampling can help improve recall for underrepresented shapes.
  - Domain-Specific Knowledge: Incorporating expert knowledge about UFO shapes and relevant terminology might enhance feature selection and model training.

**Notable observations include:**

  - Shapes like "cigar" and "chevron" have high precision (0.85 and 0.88 respectively) but relatively lower recall (0.46 and 0.33 respectively), indicating that while the model is good at predicting these shapes correctly when it does, it misses many actual instances of these shapes.
  - "Light" and "disk" shapes show a better balance between precision and recall, with "light" having the highest recall at 0.71, which suggests the model identifies a large proportion of actual light instances.
  - Shapes such as "cross" and "cone" show lower performance in both precision and recall, suggesting these shapes are more challenging for the model to classify accurately.

Overall, while the logistic regression model provides a reasonable starting point, exploring these improvement strategies can help enhance the classification of UFO shapes based on comment data.

![NLP classification results heatmap](./Images/nlp%20classification%20results%20heatmap.png)

The heatmap visualization of the classification report metrics provides a clear, visual representation of the model's performance across different shapes. This evaluation is important as it helps identify the strengths and weaknesses of the model, guiding further improvements such as more sophisticated preprocessing, feature selection, or trying different models to better capture the nuances in the data.


**Chi2 score**

The chi2 function from sklearn.feature_selection is used to calculate the chi-square statistic for each token (word) in the training data (X_train_tfidf) with respect to each class (UFO shape) in the target variable (y_train_nlp). The chi-square test is a statistical test used to determine if there is a significant association between two categorical variables. In this context, it is used to evaluate the importance of each token for distinguishing between different UFO shapes. The chi-square scores in this context measure how strongly each token (word) is associated with the different UFO shapes. A higher chi-square score indicates that the token is more important for distinguishing between the classes. The results show the top tokens based on their chi-square scores:

![Chi2 Score](./Images/Chi2%20score.png)

- Relevance and Importance

  1. Feature Importance:
    - The chi-square scores provide insight into which words (tokens) are most indicative of specific UFO shapes. For example, the high score for "diamond" indicates that the presence of the word "diamond" in the text is strongly associated with the "diamond" UFO shape.

  2. Improving Model Performance:
    - By identifying the most relevant tokens, feature selection can be used to reduce the dimensionality of the data, which can lead to improved model performance. This is because it helps the model focus on the most informative features and reduces noise from less important tokens.

  3. Interpretability:
    - Understanding which words have the highest chi-square scores helps in interpreting the model. It provides a clear indication of why the model is making certain predictions, which is crucial for transparency and trust in the model's decisions.

  4. Relevance in Classification:
    - These scores help in refining the feature set used for training the model. By selecting features with high chi-square scores, the model can achieve better precision, recall, and f1-scores, as it is trained on the most relevant data.

### Model Use and Prediction Functionality

![Sample Model](./Images/sample%20model.png)

This sample model for UFO classification or prediction functionality begins by taking an input comment describing a UFO sighting, such as "I saw a round looking thing flying across the town." The comment undergoes preprocessing, including lowercasing, punctuation removal, tokenization, stop word removal, and possibly lemmatization or stemming. This prepares the text for vectorization using a TF-IDF approach, which converts it into a numerical format understandable by machine learning models. 

The preprocessed comment is then fed into a logistic regression model trained on UFO shape data. The model predicts the most likely UFO shape category that the input comment corresponds to, such as "disk". This prediction is decoded using a LabelEncoder to its original categorical form and then outputted as "Predicted UFO shape: ['disk']".

Beyond prediction, the model serves a crucial role in testing and validating its performance on new, unseen data, ensuring it can generalize well to real-world examples of UFO sightings. In practical applications, this functionality could integrate into tools for reporting UFO sightings, providing immediate classification of UFO shapes based on user descriptions. This enhances user experience, particularly for researchers and enthusiasts who seek efficient categorization of sightings. Continuous improvement is key to enhancing the model's performance. Strategies like refining text preprocessing, experimenting with different models and parameters, augmenting training data, and integrating advanced NLP techniques could further boost accuracy and reliability. By evolving in these ways, the model aims to become a robust tool for analyzing and categorizing UFO sightings based on textual descriptions.


## Next Steps

To further enhance this UFO shape classification model and potentially extend its capabilities towards applications like chatbots, we will need to consider the following. 

1. Advanced Text Preprocessing:
  - Explore more sophisticated techniques for text preprocessing. This could involve experimenting with different tokenization methods, handling of rare words or misspellings, and incorporating more advanced linguistic features like part-of-speech tagging or named entity recognition (NER). These enhancements can improve the model's ability to extract meaningful information from diverse descriptions of UFO sightings.

2. Feature Engineering:
  - Dive deeper into feature engineering by creating additional features derived from text data. This might include sentiment analysis of comments (positive, negative, neutral), identifying specific UFO-related terms or entities (e.g., "extraterrestrial," "alien"), or capturing temporal aspects (e.g., time of sighting). Such features can enrich the model's understanding and prediction capabilities.

3. Data Augmentation and Collection:
  - Augment your dataset by collecting more diverse and extensive UFO sighting descriptions. This can involve sourcing data from different geographical locations, varying time periods, or focusing on specific types of sightings (e.g., night-time sightings, rural areas). More comprehensive data can improve model generalization and robustness.

4. Model Selection and Ensemble Techniques:
  - Consider exploring ensemble techniques or more complex models beyond logistic regression, such as Random Forests, Gradient Boosting Machines (GBM), or deep learning architectures like recurrent neural networks (RNNs) or transformers. These models can capture non-linear relationships and dependencies in the data, potentially leading to higher predictive accuracy.

5. Integration with External APIs and Tools:
  - Integrate your model with external APIs or tools like ChatGPT for a broader application scope. This could involve developing APIs that accept text inputs (UFO sighting descriptions) and return predicted UFO shapes. Seamless integration with existing platforms can enhance usability and accessibility for end-users.


