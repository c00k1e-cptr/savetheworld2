## Savetheworld2 project

By: [Jerome](https://github.com/jeromepalayoor), [Chuong](https://github.com/hollowcrust), [Daryl](https://github.com/cutekittens123)


SQL structure:

```
USERS <TABLE>
- PRIMARY KEY INT ID
- TEXT USERNAME
- TEXT PASSWORD
```


### Problem:
People do not know what disease they have because different diseases may have the same symptoms. Some diseases also take time for the symptoms to incubate in the victim, making it even harder to diagnose the user. (Also might be too late for the user if contracted a deadly disease). Less developed countries face diseases regularly, and they do not have the budget to diagnose them/ visit the doctor. Every year 6 mil people in less developed countries die from lack of healthcare, due to high demand and cost. (as well as lack of manpower)

### AI Concept:
Disease prediction tool: Use AI to predict what disease the user might have. In less developed countries (mainly Africa), can get data from them about their usual diseases (main target for the AI), as well as find nearest hospital in the area if there is a serious disease (using geography AI). AI can be beneficial to these countries which lack manpower and economy to have sufficient healthcare

There is tremendous promise in the possibilities that AI offers in transforming and improving healthcare in low-resource areas like Africa. The existing use cases show that it is a viable tool for tackling health challenges, reducing costs, and improving health access and quality. Rather than mere enthusiasm to try out new methods, an evidence-based approach should be employed in decision-making and implementation of AI in healthcare. A major lesson from the experience of AI professionals working in resource-poor settings is that AI implementation should focus on building intelligence into existing systems and institutions rather than attempting to start from scratch or hoping to replace existing systems. African countries must also enact laws and policies that will guide the application of this technology to healthcare and protect the users.

### Use of AI
- Ask the user for the symptoms, how they are feeling. Another idea is to use voice to talk to the AI doctor, since how the user sound may have correlations to certain diseases (Eg stroke and paralysis)
- The user might also use photographs to show rashes and spots for other diseases, or taking picture of the throat
- Use AI to search in database for corresponding diseases, and find nearest hospital in the area if there is a serious disease (using google maps)
- AI can generate more feedback and advice for the user (making it more personalised)

### Impact
Reduce damage diseases has done, especially in less developed countries. It also helps to relieve stresses that the user might have since they might think that they have contracted a deadly virus. It is also more cost efficient to use AI rather than consulting doctor, saving money and lives of those who can't afford it.

### Ethics
Using AI for human health might not be optimal for some users since they might fear trusting and risking their health to AI. To have an effective AI, medical information of patients may need to be used for better analysis.

### Cybersecurity
Cybersecurity is essential in our application to protect the confidentiality, integrity, and availability of sensitive data in our application which can be targeted by cybercriminals seeking to steal, manipulate, or destroy the data. For example, our web app will have an SQL backend, hence we need to use sanitization methods to prevent SQL injection and other possible attacks.


### Credits
- [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120372/)
- [frontiersin.org](https://www.frontiersin.org/articles/10.3389/fdgth.2020.00006/full)