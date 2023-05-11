## Savetheworld2 project

### Problem:
Diabetes is chronic disease and have long term health risks on the victim if not treated early enough. Diabetes can lead to various complications if not well managed. Countries around the world face diabetes as a regular issue, and some do not have the budget to diagnose them/visit the doctor for appropriate advice. According to the International Diabetes Federation (IDF), in 2021, approximately 540 million adults (age 20-79) had diabetes worldwide, which represents around 10.5% of the global adult population. Diabetes cases has been on the rise over the years, these countries have the highest percentage of obesity in the country


### AI Concept:
Diabetes prediction tool: Use AI to predict diabetes the user might have. In countries all over the world, we can get data from them about their current health state and lifestyle, as well as their geographical location to scout for hospitals in the area and provide personalised recommendations (using geography and text AI). The AI can be beneficial to countries which are in severe risk at diabetes, lowering its risk on victims. The AI can be used to reduce the risk of diabetes by enabling early detection, personalised risk assessments, and lifestyle recommendations. By analyzing various data sources, AI algorithms can identify individuals at a higher risk of developing diabetes and provide tailored interventions. Additionally, predictive analytics of diabetes in users can be used, as well as virtual assistants and chatbots in the form can offer recommendations to the user.


### Use of AI
- Ask the user for, HBa1c (haemoglobin) and blood glucose levels,
- Use AI to classify (using diabetes prediction dataset from Kaggle) for diagnosing user, as well as corresponding recommendations. It also can be used to find nearest hospital in the area if there is an emergency (using google maps)
- Review systems to have more feedback for AI so that it can also improve


### Impact
Reduce damage diabetes has done. It also helps to relieve stresses that the user might have since 1 out of 5 people (Source: Centre for disease control and prevention), from the 10% who suffer from diabetes, know that they have diabetes. It is also more cost and time efficient to use AI rather than consulting doctor, saving money and lives of those who can't afford it.


### Ethics
Using AI for human health might not be optimal for some users since they might fear trusting and risking their health to AI. To have an effective AI, medical information of patients may need to be used for better analysis.

### Cybersecurity
Cybersecurity is essential in our application to protect the confidentiality, integrity, and availability of sensitive data in our application which can be targeted by cybercriminals seeking to steal, manipulate, or destroy the data. For example, our web app will have an SQL backend, hence we need to use sanitization methods to prevent SQL injection and other possible attacks such as Server Side Template Injection (SSTI).

SQL structure:
```
users <TABLE>
- id INTEGER PRIMARY KEY AUTOINCREMENT
- username TEXT
- password TEXT
- first_name TEXT
- last_name TEXT
- age INTEGER
- height INTEGER
- weight INTEGER
- gender FLOAT

results <TABLE>
- id INTEGER PRIMARY KEY AUTOINCREMENT
- username TEXT FOREIGN KEY
- test_name TEXT
- result FLOAT
- date TEXT
```

By: [Jerome](https://github.com/jeromepalayoor), [Chuong](https://github.com/hollowcrust) and [Daryl](https://github.com/cutekittens123) of Class 24/14 ASRJC

### Credits
- [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120372/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120372/)
- [https://www.frontiersin.org/articles/10.3389/fdgth.2020.00006/full](https://www.frontiersin.org/articles/10.3389/fdgth.2020.00006/full)
- [https://www.cdc.gov/diabetes/library/spotlights/diabetes-facts-stats.html](https://www.cdc.gov/diabetes/library/spotlights/diabetes-facts-stats.html)
- [iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- [ML SKlearn model bruteforce code](https://github.com/beanbeah/ML/blob/main/sklearn-ml-bruteforce.py) by [Sean](https://github.com/beanbeah)